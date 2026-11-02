"""The professor-facing workflow: dated headlines, one forecast, saved results."""
from __future__ import annotations

import html
import io
import json
import zipfile
from collections import Counter
from datetime import date, timedelta

import pandas as pd
import psutil
import streamlit as st

from .core import Request, digest, now
from .experiments import (EASTERN, WINDOWS, ExperimentStore, bundle, collector_headlines,
                          create_study, packet, parse_headlines, report, save_labels, trading_dates)
from .experiment_worker import alive, pause, start
from .gemma import credential
from .query import get_news, manifest, serialize
from .sources import REGISTRY
from .worker import enqueue, start_worker


def job_list(store, ticker):
    return [r for r in store.rows("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200")
            if ticker in json.loads(r["request"]).get("tickers", [])]


def state_key(store, ticker, name):
    return f"ws_{digest(str(store.settings.root))[:10]}_{ticker}_{name}"


def go(page):
    st.session_state["main_page"] = page


def day_label(day):
    return date.fromisoformat(day).strftime("%d %b %Y")


def job_label(job):
    spec = json.loads(job["request"])
    return f"{day_label(spec['since'])} – {day_label(spec['until'])} · collected {job['created_at'][:16].replace('T', ' ')} UTC"


def current_results(store, ticker, job_id, *, related=True):
    job = store.job(job_id)
    spec = json.loads(job["request"])
    return get_news(store, ticker, now(), since=spec["since"], until=spec["until"], job_id=job_id,
                    scopes=["company", "sector"] if related else ["company"], include_price_recaps=False)


def headline_table(records, *, page_key=None, publisher=True, newest_first=True):
    if not records:
        return
    ordered = sorted(records, key=lambda r: (r.get("publication_date") or r.get("date", ""), r["headline"]), reverse=newest_first)
    if page_key and len(ordered) > 30:
        page = st.selectbox("Headline page", range((len(ordered) + 29) // 30),
                            format_func=lambda x: f"{x*30+1}–{min((x+1)*30,len(ordered))} of {len(ordered)}", key=page_key)
        ordered = ordered[page*30:(page+1)*30]
    rows = []
    for r in ordered:
        day = r.get("publication_date") or r.get("date", "")
        title = html.escape(r["headline"])
        url = r.get("url", "")
        if url.startswith(("https://", "http://")):
            title = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        byline = ''
        if publisher:
            kind = "Company announcement" if r.get("source_role") == "issuer" else "Policy notice" if r.get("kind") == "regulatory_notice" else "Related context" if r.get("relevance_scope") == "sector" else "Company news"
            suffix = f" · {kind}" if kind else ""
            byline = f'<div class="news-byline">{html.escape(r.get("publisher", ""))}{html.escape(suffix)}</div>'
        rows.append(f'<tr><td class="news-date">{html.escape(day_label(day))}</td><td>{title}{byline}</td></tr>')
    st.markdown('<div class="news-table-wrap"><table class="news-table"><thead><tr><th>Published</th><th>Headline</th></tr></thead><tbody>'
                + ''.join(rows) + '</tbody></table></div>', unsafe_allow_html=True)


def _new_search(key):
    st.session_state[key] = None


def _to_predict(store, ticker, job, records, related):
    spec = json.loads(job["request"])
    from .core import _calendar
    target = _calendar("XNYS").date_to_session(str(date.fromisoformat(spec["until"]) + timedelta(days=1)), direction="next").date()
    st.session_state[state_key(store, ticker, "prediction_search")] = job["id"]
    st.session_state[state_key(store, ticker, "selected_job")] = job["id"]
    st.session_state[state_key(store, ticker, "target")] = target
    span = (date.fromisoformat(spec["until"]) - date.fromisoformat(spec["since"])).days + 1
    st.session_state[state_key(store, ticker, "window")] = max((w for w in WINDOWS if w <= span), default=5)
    st.session_state[state_key(store, ticker, "prediction_related")] = related
    st.session_state[state_key(store, ticker, "selected_ids")] = [r["record_id"] for r in records]
    go("Predict")


def headlines(store, ticker, workspace):
    st.title("Find your headlines")
    st.write(f"Search news for **{ticker}**, review the dates, then use it for a prediction.")
    jobs = job_list(store, ticker)
    selected_key = state_key(store, ticker, "search")
    if selected_key not in st.session_state:
        st.session_state[selected_key] = jobs[0]["id"] if jobs else None
    last = json.loads(jobs[0]["request"]) if jobs else {}
    today = pd.Timestamp.now(tz=EASTERN).date()
    company = store.settings.companies()[ticker]
    defaults = ["news_sites", "archives", "aggregators"]
    defaults += ["publishers"] if company.source_ids else []
    defaults += ["issuer"] if company.issuer_index else []
    defaults += ["cloud"] if ticker == "BABA" else []
    defaults += ["feeds"] if company.feeds else []
    with st.expander("Search again · change dates or sources", expanded=not jobs or st.session_state[selected_key] is None):
        with st.form("headline_search_" + ticker, border=False):
            a, b = st.columns(2)
            since = a.date_input("News published from", date.fromisoformat(last["since"]) if last else today-timedelta(days=20))
            until = b.date_input("Through", date.fromisoformat(last["until"]) if last else today-timedelta(days=1))
            st.caption("Both dates included. Exact timestamps use New York dates; date-only source dates stay as reported. Headlines only.")
            with st.expander("Choose sources & collection settings"):
                options = [s for s in REGISTRY if s not in ("social", "context")]
                chosen = st.multiselect("Sources to search", options, default=defaults, format_func=lambda x: REGISTRY[x]["name"])
                st.caption("CNBC and TechNode provide reporting; Alibaba sources provide company announcements. Recent feeds may have little historical coverage. Policy notices are optional.")
                if any(s in chosen for s in ("sec", "sec_news")) and not store.settings.contact:
                    st.warning("SEC needs a real contact email in Settings → News sources & connections. No account or API key is required.")
                pages = st.selectbox("Archive search depth", [15, 25], index=1, format_func=lambda x: f"Up to {x} archive pages per source")
                ongoing = st.checkbox("Keep checking every 15 minutes", value=False)
            submit = st.form_submit_button("Find headlines", type="primary", disabled=workspace == "Offline demo", width="stretch")
        if submit:
            active = next((j for j in jobs if j["status"] in ("queued", "running") and not j["cancel"]), None)
            if active:
                st.session_state[selected_key] = active["id"]
                st.info("A search is already running. Its results appear below.")
            else:
                try:
                    key = enqueue(store, Request(tickers=[ticker], sources=chosen, since=since, until=until,
                                  collect_bodies=False, discovery_pages=pages, mode="watch" if ongoing else "once"))
                    start_worker(store.settings, once=not ongoing)
                    st.session_state[selected_key] = key
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
    if workspace == "Offline demo":
        st.info("Synthetic demonstration headlines. Live searches and model calls are disabled here.")
    jobs = job_list(store, ticker)
    lookup = {j["id"]: j for j in jobs}
    if st.session_state[selected_key] not in lookup:
        st.session_state[selected_key] = None
    a, b = st.columns([4, 1])
    job_id = a.selectbox("Saved search", [None, *lookup], key=selected_key,
                        format_func=lambda x: "New search · no results selected" if x is None else job_label(lookup[x]))
    b.write("")
    b.button("Clear view", on_click=_new_search, args=(selected_key,), help="Hide this result list. Saved searches and predictions are preserved.", width="stretch")
    if not job_id:
        st.info("Choose dates above and press Find headlines. Previous searches remain in Saved search.")
        return
    collection_panel(store.settings.root, ticker, job_id)


@st.fragment(run_every=3)
def collection_panel(home, ticker, job_id):
    from .core import Settings
    from .store import Store
    store = Store(Settings(home))
    job = store.job(job_id)
    spec = json.loads(job["request"])
    busy = job["status"] in ("queued", "running")
    workers = store.rows("SELECT pid FROM worker_lease")
    worker_pids = [job.get("owner_pid"), *(w["pid"] for w in workers)]
    old_queue = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(job["created_at"])).total_seconds() > 20
    interrupted = busy and (job["status"] == "running" or old_queue) and not any(p and psutil.pid_exists(p) for p in worker_pids)
    if interrupted:
        busy = False
    labels = {"queued": "Waiting to start", "running": "Searching sources…", "partial": "Search finished · some coverage gaps",
              "completed": "Search finished", "cancelled": "Search stopped", "failed": "Search paused by an error"}
    status = "Search interrupted · saved headlines preserved" if interrupted else labels.get(job["status"], job["status"])
    st.markdown(f'<div class="search-state">{"<span class=live-dot></span>" if busy else ""}{status}</div>', unsafe_allow_html=True)
    st.subheader(f"{ticker} · {day_label(spec['since'])} – {day_label(spec['until'])}")
    st.caption(f"Only headlines from this search and these publication dates appear below. Collected here: {job['created_at'][:16].replace('T',' ')} UTC.")
    if busy:
        a, b = st.columns([3, 1])
        events = store.rows("SELECT message FROM events WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,))
        a.caption(events[0]["message"] if events else "Starting…")
        if b.button("Stop search", key="stop_search_" + job_id, disabled=bool(job["cancel"]), width="stretch"):
            store.execute("UPDATE jobs SET cancel=1,next_run=NULL,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?", (job_id,))
            st.rerun(scope="fragment")
        if job["cancel"]:
            st.info("Stopping safely. Headlines already found will be kept.")
    elif interrupted or job["status"] in ("cancelled", "failed"):
        if st.button("Resume this search", key="resume_search_" + job_id):
            store.execute("UPDATE jobs SET status='queued',cancel=0,error=NULL WHERE id=?", (job_id,))
            start_worker(store.settings, once=spec.get("mode") != "watch")
            st.rerun(scope="fragment")
    if job.get("next_run"):
        st.caption("Scheduled checks are enabled for this ticker.")
        if st.button("Stop scheduled checks", key="stop_schedule_" + job_id):
            for item in job_list(store, ticker):
                if item.get("next_run") or item["status"] in ("queued", "running"):
                    store.execute("UPDATE jobs SET next_run=NULL,cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?", (item["id"],))
            st.rerun(scope="fragment")
    related = st.checkbox("Include related China / industry news", value=True, key="scope_" + job_id)
    result = current_results(store, ticker, job_id, related=related)
    records = result["records"]
    text = st.text_input("Filter these headlines", placeholder="Search a headline or publisher", key="search_text_" + job_id)
    if text:
        records = [r for r in records if text.casefold() in (r["headline"] + " " + r["publisher"]).casefold()]
    st.markdown(f'<div class="result-summary"><strong>{len(records)}</strong> headlines <span>·</span> <strong>{len({r["publisher_group"] for r in records})}</strong> publisher groups <span>·</span> <strong>{len({r["publication_date"] for r in records})}</strong> publication dates</div>', unsafe_allow_html=True)
    if records:
        a, b = st.columns([2, 1])
        if a.button("Use these headlines →", type="primary", key="use_" + job_id, on_click=_to_predict,
                    args=(store, ticker, job, records, related), width="stretch"):
            st.rerun(scope="app")
        rows = [{"date": r["publication_date"], "headline": r["headline"]} for r in sorted(records, key=lambda r:r["publication_date"])]
        b.download_button("Download headlines", serialize(rows, "csv"), file_name=f"{ticker}-{spec['since']}-{spec['until']}-headlines.csv",
                          mime="text/csv", key="csv_" + job_id, width="stretch", on_click="ignore")
        headline_table(records, page_key="news_page_" + job_id)
    elif busy:
        st.info("Searching… No matching headlines in your dates yet. Results appear as sources respond.")
    else:
        st.info("No usable headlines in this selection. Try another date range or source. News outside your chosen dates is not shown.")
    with st.expander("Source coverage & search details"):
        st.caption("Counts below include everything a source returned, including items rejected as unrelated or outside your dates. They are not extra headlines in your result list.")
        runs = store.rows("SELECT source,status,coverage,detail,discovered FROM source_runs WHERE job_id=? AND ticker=? ORDER BY id", (job_id,ticker))
        for r in runs:
            st.write(f"**{REGISTRY.get(r['source'], {}).get('name', r['source'])}** · {r['status']} · {r['discovered']} items returned")
            st.caption(r["detail"] or r["coverage"])
        reasons = store.rows("SELECT reason,COUNT(*) n FROM discoveries WHERE job_id=? AND ticker=? AND state='rejected' GROUP BY reason", (job_id,ticker))
        if reasons:
            st.dataframe(reasons, hide_index=True, width="stretch")
        st.caption("Search reference: " + job_id + ". Public archives are incomplete; a publication date alone does not certify an unchanged historical headline.")
        if records:
            payload = serialize(records, "jsonl")
            buf = io.BytesIO()
            selected_result = {**result, "records": records}
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("headlines.jsonl", payload)
                z.writestr("manifest.json", json.dumps(manifest(selected_result,payload,"jsonl"),indent=2))
            st.download_button("Download source evidence & manifest", buf.getvalue(), file_name=f"{ticker}-{job_id}-evidence.zip", key="evidence_" + job_id,on_click="ignore")


def predict(store, ticker, workspace):
    st.title("Make a prediction")
    st.write(f"Will **{ticker} close higher or lower than it opens** on your chosen trading date?")
    st.caption("Gemma 4 31B · dated headlines only · no tools or web search in model requests")
    if workspace == "Offline demo":
        st.info("Model calls are disabled in the synthetic demo. Switch to Live collection to predict.")
        return
    db = ExperimentStore(store.settings.root)
    try:
        _prediction_form(store, db, ticker)
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(str(exc))


def _prediction_form(store, db, ticker):
    source = st.radio("Headline input", ["Saved search", "Paste / upload headlines"], horizontal=True)
    records, strict, chosen = [], False, None
    if source == "Saved search":
        jobs = job_list(store, ticker)
        if not jobs:
            st.info("No saved headlines yet. Find headlines first, or paste your own dated headlines.")
            st.button("Find headlines →", on_click=go, args=("Headlines",))
            return
        ids = [j["id"] for j in jobs]
        job_key = state_key(store,ticker,"prediction_search")
        if st.session_state.get(job_key) not in ids:
            st.session_state[job_key] = ids[0]
        chosen = st.selectbox("Use headlines from", ids, key=job_key, format_func=lambda x:job_label(next(j for j in jobs if j["id"]==x)))
        spec = json.loads(store.job(chosen)["request"])
        with st.expander("Input options"):
            related_key = state_key(store,ticker,"prediction_related")
            if related_key not in st.session_state:
                st.session_state[related_key] = True
            related = st.checkbox("Include related China / industry headlines", key=related_key)
            strict = st.checkbox("Require news actually observed before the historical cutoff", value=False,
                                 help="For forward collection. Historical news collected today cannot pass a past observation cutoff.")
        records, _ = collector_headlines(store,ticker,strict=strict,scopes=["company","sector"] if related else ["company"],
                                        job_id=chosen,since=spec["since"],until=spec["until"])
        selected_ids = st.session_state.get(state_key(store,ticker,"selected_ids"))
        if selected_ids is not None and chosen == st.session_state.get(state_key(store,ticker,"selected_job")):
            # The review-page button can pass a filtered selection; changing searches resets it.
            subset = [r for r in records if r["record_id"] in selected_ids]
            if len(subset) < len(records):
                st.caption(f"Using {len(subset)} headlines selected on the Headlines page.")
                if st.button("Use all headlines from this search"):
                    st.session_state.pop(state_key(store,ticker,"selected_ids"),None)
                    st.rerun()
            records = subset
    else:
        upload = st.file_uploader("Upload CSV with date,headline", type=["csv"])
        text = st.text_area("Paste dated headlines", placeholder="2025-10-16 -- Alibaba announces a new partnership", height=120)
        records = parse_headlines(upload.getvalue().decode("utf-8-sig") if upload else text)
    mode = st.radio("Prediction mode", ["One date", "Compare dates"], horizontal=True)
    from .core import _calendar
    target_key = state_key(store,ticker,"target")
    if target_key not in st.session_state:
        if chosen:
            last = date.fromisoformat(json.loads(store.job(chosen)["request"])["until"])
            default = _calendar("XNYS").date_to_session(str(last + timedelta(days=1)), direction="next").date()
        else:
            default = _calendar("XNYS").date_to_session(str(pd.Timestamp.now(tz=EASTERN).date()), direction="next").date()
        st.session_state[target_key] = default
    a,b = st.columns(2)
    if mode == "One date":
        target = a.date_input("Prediction date", key=target_key)
        windows = [b.selectbox("News window", WINDOWS, key=state_key(store,ticker,"window"), format_func=lambda w:f"Previous {w} calendar days")]
        targets = [str(target)]
    else:
        first = a.date_input("First prediction date", value=st.session_state[target_key])
        last = b.date_input("Last prediction date", value=first)
        targets = trading_dates(first,last)
        windows = st.multiselect("Compare news windows", WINDOWS, default=WINDOWS, format_func=lambda w:f"{w} days")
    repeats = 1
    if mode != "One date":
        with st.expander("Repeat each prediction (optional)"):
            repeats=st.number_input("Tries per date/window",1,3,1)
            st.caption("Extra tries measure consistency. Accuracy counts the first try per date/window.")
    packets = [packet(records,ticker,d,w,strict=strict) for d in targets for w in windows]
    nonempty = [p for p in packets if p["headlines"]]
    if packets:
        ix = st.selectbox("Preview prediction input",range(len(packets)),format_func=lambda i:f"{packets[i]['target']} · {packets[i]['window']} days · {len(packets[i]['headlines'])} headlines") if len(packets)>1 else 0
        p=packets[ix]
        count = len(p["headlines"])
        st.markdown(f"### {count} {'headline' if count == 1 else 'headlines'} for this prediction")
        st.caption(f"{day_label(p['since'])} – {day_label(str(date.fromisoformat(p['target'])-timedelta(days=1)))} · no target-day news · weekends included")
    if not nonempty:
        st.info("No headlines in these prediction windows. Choose another saved search/date, or collect the preceding days first.")
    try:
        credential(db.home)
        configured=True
    except (ValueError,OSError):
        configured=False
        st.info("Save your Google API key in Settings → Model connection to enable predictions.")
    in_flight = any(alive(s["pid"]) for s in db.rows("SELECT pid FROM studies WHERE status IN ('running','starting')"))
    if st.button("Predict UP / DOWN" if mode=="One date" else f"Run {len(nonempty)*repeats} predictions",type="primary",disabled=not nonempty or not configured or in_flight,width="stretch"):
        key=create_study(db,ticker,records,targets,windows,repeats=repeats,strict=strict)
        st.session_state[state_key(store,ticker,"active_prediction")]=key
        start(db,key)
    active=st.session_state.get(state_key(store,ticker,"active_prediction"))
    if active:
        prediction_progress(db.home,active)
    if packets:
        headline_table(p["headlines"],page_key="prediction_packet_page",publisher=False,newest_first=False)
        with st.expander("Exact model request & timing details"):
            st.text_area("Exact dated-headline input",p["text"],height=220,disabled=True)
            st.caption("Only ticker/target instructions and these date/headline pairs are sent. Prices, URLs, source notes and previous conversations are excluded. Historical dates do not certify unchanged text or absence of training contamination.")


@st.fragment(run_every=3)
def prediction_progress(home, study_id):
    db=ExperimentStore(home)
    study=db.study(study_id)
    rows=db.rows("SELECT * FROM predictions WHERE study_id=? ORDER BY target,window,trial",(study_id,))
    done=[r for r in rows if r["status"]=="completed"]
    labels={"ready":"Ready","starting":"Starting prediction…","running":"Asking Gemma…","paused":"Prediction paused","stopped":"Prediction stopped","completed":"Prediction saved"}
    st.subheader(labels.get(study["status"],study["status"]))
    if study["status"] in ("starting","running"):
        st.progress(len(done)/len(rows) if rows else 0,text=f"{len(done)} of {len(rows)} predictions saved")
    if study["error"]:
        st.warning(study["error"])
    if study["status"] != "completed":
        a,b,c=st.columns(3)
        if a.button("Pause",key="ws_pause_"+study_id,disabled=study["status"] not in ("starting","running")):
            pause(db,study_id);st.rerun(scope="fragment")
        if b.button("Resume",key="ws_resume_"+study_id,disabled=alive(study["pid"])):
            try:start(db,study_id);st.rerun(scope="fragment")
            except ValueError as exc:st.error(str(exc))
        if c.button("Stop prediction",key="ws_stop_"+study_id,disabled=study["status"]=="stopped"):
            pause(db,study_id,stop=True);st.rerun(scope="fragment")
        st.caption("Pause/Stop prevents new calls. A call already sent may finish. Closing the browser does not stop the worker.")
    if len(rows)==1 and done:
        r=done[0]
        direction=r["direction"]
        st.markdown(f'<div class="prediction-card {direction.lower()}"><div class="prediction-label">{html.escape(study["ticker"])} · {day_label(r["target"])}</div><div class="prediction-direction">{direction}</div><div class="prediction-meaning">Predicted close {"above" if direction=="UP" else "below"} the opening price</div><p>{html.escape(r["reason"] or "")}</p></div>',unsafe_allow_html=True)
        st.caption(f"Saved automatically · {r['window']}-day headline window · actual outcome has not been supplied to the model")
    elif rows:
        st.dataframe([{"Date":r["target"],"Window":f"{r['window']} days","Try":r["trial"],"Prediction":r["direction"] or "—","Status":r["status"]} for r in rows],hide_index=True,width="stretch")


def history(store,ticker,workspace):
    st.title("Prediction history")
    st.write("Your saved UP/DOWN predictions. Actual prices are added separately, only for scoring.")
    db=ExperimentStore(store.settings.root)
    studies=db.rows("SELECT * FROM studies WHERE ticker=? ORDER BY created_at DESC",(ticker,))
    if not studies:
        st.info("No predictions saved yet. Find headlines, then open Predict.")
        st.button("Find headlines →",on_click=go,args=("Headlines",))
        return
    def label(s):
        spec=json.loads(s["spec"])
        dates=spec["targets"]
        return f"{dates[0]}"+(f" – {dates[-1]}" if len(dates)>1 else "")+f" · {'/'.join(map(str,spec['calendar_days']))}-day inputs · {s['status']} · saved {s['created_at'][:16]} UTC"
    ids=[s["id"] for s in studies]
    selected=st.selectbox("Saved prediction run",ids,format_func=lambda x:label(next(s for s in studies if s["id"]==x)))
    prediction_progress(db.home,selected)
    result=report(db,selected)
    spec=json.loads(db.study(selected)["spec"])
    with st.expander("Check accuracy with actual prices (optional)"):
        st.caption("Use independently checked prices for BABA’s US-listed shares. These prices never enter model requests.")
        if len(spec["targets"])==1:
            with st.form("ws_score_" + selected):
                a,b=st.columns(2)
                opening=a.number_input("Actual opening price",min_value=0.0,value=0.0,step=0.01,key="ws_open_"+selected)
                closing=b.number_input("Actual closing price",min_value=0.0,value=0.0,step=0.01,key="ws_close_"+selected)
                submitted=st.form_submit_button("Save actual prices")
            if submitted:
                try:save_labels(db,selected,f"date,open,close\n{spec['targets'][0]},{opening},{closing}\n");st.rerun()
                except ValueError as exc:st.error(str(exc))
        template="date,open,close\n"+"\n".join(d+",," for d in spec["targets"])
        st.download_button("Download price template",template,file_name=f"{ticker}-prices.csv",mime="text/csv",on_click="ignore")
        upload=st.file_uploader("Upload actual prices CSV",type=["csv"],key="ws_prices_"+selected)
        if st.button("Import actual prices",disabled=upload is None):
            try:save_labels(db,selected,upload.getvalue().decode("utf-8-sig"));st.rerun()
            except (ValueError,UnicodeError) as exc:st.error(str(exc))
        if result["label_version"]:
            st.dataframe([{"Window":f"{r['window_days']} days","Dates scored":r["scored_dates"],"Correct":r["correct"],"Accuracy":"—" if r["accuracy"] is None else f"{r['accuracy']:.1%}","Always UP baseline":"—" if r["always_up_accuracy"] is None else f"{r['always_up_accuracy']:.1%}"} for r in result["summary"]],hide_index=True,width="stretch")
            st.caption("First try per date/window. Flat days are excluded from binary accuracy; missing/failed calls stay recorded. Compare windows on the same dates.")
        else:st.info("No actual prices supplied yet. Your predictions are still saved.")
    st.download_button("Download predictions & report",bundle(db,selected),file_name=f"{ticker}-predictions-{selected}.zip",mime="application/zip",type="primary",on_click="ignore")
    with st.expander("Exact inputs, responses & research details"):
        rows=db.rows("SELECT * FROM predictions WHERE study_id=? ORDER BY target,window,trial",(selected,))
        ix=st.selectbox("Inspect prediction",range(len(rows)),format_func=lambda i:f"{rows[i]['target']} · {rows[i]['window']} days · try {rows[i]['trial']}")
        r=rows[ix]
        st.text_area("Exact saved request",r["prompt"],height=240,disabled=True)
        st.code(r["raw"] or "No response saved yet",language="json")
        st.json({"returned_model_version":r["model_version"],"request_started_at":r["started_at"],"response_saved_at":r["finished_at"],"report":result})
