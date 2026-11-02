"""Local UI using the same application services as the CLI."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from newscollector.core import Company, Settings, SourceProfile, now
from newscollector.demo import load_demo
from newscollector.sources import REGISTRY
from newscollector.store import Store
from newscollector.worker import start_worker
from newscollector.quality import corpus_report, create_audit, save_audit_review, audit_report
from newscollector.model_settings import model_settings
from newscollector.experiments import ExperimentStore
from newscollector import workspace as work_ui
from importlib.resources import files

st.set_page_config(page_title="Newsroom · Research collector", page_icon="📰", layout="wide")
st.set_option("client.toolbarMode", "minimal")
st.markdown("""<style>
.block-container{padding-top:2.6rem;max-width:1280px;padding-bottom:3rem}
[data-testid=stSidebar]{background:#f4f6fa;border-right:1px solid #e2e8f0}
h1{letter-spacing:-.035em;font-size:2.35rem!important;color:#142b45}
h2{font-size:1.55rem!important} h3{font-size:1.12rem!important}
[data-testid=stMetric]{border:1px solid #dce4ed;border-radius:12px;padding:16px;background:#fff}
[data-testid=stMetricValue]{font-size:1.8rem}
[data-testid=stVerticalBlockBorderWrapper]{border-radius:12px}
[data-testid=stSidebar] [role=radiogroup]{gap:7px}
[data-testid=stSidebar] [role=radiogroup] label{padding:8px 10px;border-radius:8px;background:#fff}
[data-testid=stBaseButton-primary]{font-weight:650;min-height:44px}
button{border-radius:8px!important;min-height:38px} footer{visibility:hidden}
.st-key-live_stop button:not(:disabled){color:#a61c31;border-color:#efbdc5;background:#fff0f2;font-weight:650}
.eyebrow{font-size:.78rem;letter-spacing:.13em;color:#386089;font-weight:700;margin-bottom:14px}
.headline-card{font-size:1.08rem;font-weight:600;line-height:1.5;margin:0 0 12px;color:#172b45}
</style>""", unsafe_allow_html=True)

st.markdown("<style>" + files("newscollector.resources").joinpath("workspace.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
base = Path(os.environ.get("NEWS_COLLECTOR_HOME", "data")).resolve()
with st.sidebar:
    st.markdown("### Newsroom")
    st.caption("Headlines → UP / DOWN")
    workspace = st.selectbox("Workspace", ["Live collection", "Offline demo"])
    root = base / "demo" if workspace == "Offline demo" else base
    store = Store(Settings(root))
    if workspace == "Offline demo":
        if not store.rows("SELECT id FROM jobs WHERE json_extract(request,'$.demo_loaded')=1"):
            load_demo(store)
        st.info("Synthetic examples. No real company news.")
    companies = store.settings.companies()
    tickers = ["DEMO"] if workspace == "Offline demo" else [x for x in companies if x != "DEMO"]
    active_ticker = st.selectbox("Company / ticker", tickers, key="active_ticker", help="Select BABA for Alibaba’s US-listed shares. Add more companies in Settings.")
    if st.session_state.get("main_page") not in (None, "Headlines", "Predict", "History", "Settings"):
        st.session_state["main_page"] = "Headlines"
    main_page = st.radio("Main sections", ["Headlines", "Predict", "History", "Settings"], key="main_page", label_visibility="collapsed")
    page = main_page
    if main_page == "Settings":
        st.caption("Connections and optional research tools")
        page = st.selectbox("Settings & research tools", ["Model", "Sources", "Companies", "Overview", "Corpus quality", "Audit"],
                            format_func=lambda x: {"Model":"Model connection", "Sources": "News sources & connections", "Companies": "Company profiles", "Corpus quality": "Coverage diagnostics", "Audit": "Relevance review", "Overview": "Run diagnostics"}[x])
    st.divider()
    st.caption("Free news collection · Local storage")
    st.caption("Gemma receives dated headlines only when you press Predict. No tools or search in model requests.")
    st.button("Refresh page", on_click=lambda: None, width="stretch")

st.markdown('<div class="eyebrow">NEWSROOM / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
if page == "Headlines":
    work_ui.headlines(store, active_ticker, workspace)
elif page == "Predict":
    work_ui.predict(store, active_ticker, workspace)
elif page == "History":
    work_ui.history(store, active_ticker, workspace)
elif page == "Model":
    st.title("Model connection")
    model_settings(ExperimentStore(store.settings.root))



def show_events(job_id):
    events = store.rows("SELECT at,level,stage,message FROM events WHERE job_id=? ORDER BY id DESC LIMIT 50", (job_id,))
    if events:
        st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")


def overview():
    st.title("Your news collection")
    st.markdown('<p class="intro">Follow the collection, inspect what was saved, and prepare a traceable dataset.</p>', unsafe_allow_html=True)
    counts = {r["state"]: r["n"] for r in store.rows("SELECT state,COUNT(*) n FROM discoveries GROUP BY state")}
    cols = st.columns(4)
    for col, title, value in zip(cols, ["Discovered", "Accepted", "Rejected", "Pending"],
                                [sum(counts.values()), counts.get("accepted", 0), counts.get("rejected", 0), counts.get("pending", 0) + counts.get("fetching", 0)]):
        col.metric(title, value)
    st.caption("Counts include repeated observations. Exports select document versions and can suppress exact text duplicates.")
    st.caption("Accepted records the original collection decision. Current exports also apply scope, time and recorded content-quarantine rules.")
    jobs = store.rows("SELECT id,status,created_at,started_at,finished_at,next_run,error FROM jobs ORDER BY created_at DESC LIMIT 25")
    if not jobs:
        st.info("Start in Headlines, or switch to Offline demo to explore the complete workflow.")
    else:
        st.subheader("Recent collection runs")
        st.dataframe(pd.DataFrame(jobs), hide_index=True, width="stretch")
        job_id = st.selectbox("Inspect run", [x["id"] for x in jobs])
        a, b, c = st.columns(3)
        if a.button("Stop run"):
            store.execute("UPDATE jobs SET cancel=1,next_run=NULL WHERE id=?", (job_id,))
            st.success("Stop requested. The worker finishes its safe checkpoint.")
        if b.button("Resume"):
            store.execute("UPDATE jobs SET status='queued',cancel=0,error=NULL WHERE id=?", (job_id,))
            start_worker(store.settings)
            st.success("Resume queued. Existing versions are preserved.")
        if c.button("Recover worker"):
            start_worker(store.settings)
            st.success("Worker started or an existing worker retained its lease.")
        st.subheader("What happened")
        show_events(job_id)
    left, right = st.columns(2)
    with left:
        st.subheader("Observed activity by UTC date")
        activity = store.rows("SELECT substr(first_seen,1,10) day,COUNT(*) observations FROM discoveries GROUP BY day ORDER BY day")
        if activity:
            st.bar_chart(pd.DataFrame(activity).set_index("day"))
        st.caption("Empty dates can mean no collection. This chart does not claim there was no news.")
    with right:
        st.subheader("Why items were excluded")
        reasons = store.rows("SELECT reason,COUNT(*) count FROM discoveries WHERE state='rejected' GROUP BY reason ORDER BY count DESC")
        if reasons:
            st.dataframe(pd.DataFrame(reasons), hide_index=True, width="stretch")

if page == "Overview":
    st.fragment(run_every=3)(overview)()

elif page == "Sources":
    st.title("Source health & coverage")
    st.caption("All built-in collection paths are account-free. SEC asks for a real contact email in the request header; it does not require an account or API key.")
    st.caption("Public social posts are optional and collected separately from news headlines. The BABA preset pins CNBC’s verified Bluesky identity. X collection is not implemented.")
    with st.expander("SEC contact setting"):
        st.caption("SEC requests identify this collector and your real contact email. This setting is saved locally, outside Git.")
        contact = st.text_input("Contact email", value=store.settings.contact)
        if st.button("Save SEC contact"):
            try:
                store.settings.save_contact(contact)
                st.success("Contact saved. New collection workers will use this setting.")
            except ValueError as exc:
                st.error(str(exc))
    runs = store.rows("SELECT ticker,source,status,started_at,finished_at,discovered,coverage,detail FROM source_runs ORDER BY id DESC LIMIT 200")
    if runs:
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    else:
        st.info("Source runs will appear here after the first collection.")
    st.caption("A successful request does not imply exhaustive coverage. Unknown and missing intervals stay unknown.")
    st.subheader("Publisher and context source profiles")
    profiles = store.settings.source_profiles()
    st.dataframe(pd.DataFrame([{"id": key, "name": p.name, "scope": p.scope, "role": p.role,
                               "body_mode": p.body_mode, "publisher_group": p.publisher_group,
                               "rights_note": p.rights_note} for key, p in profiles.items()]), hide_index=True, width="stretch")
    st.caption("Global Voices provides sparse independent China reporting/analysis, not comprehensive financial-wire coverage. Fed/BLS are primary US context, not independent company news.")
    with st.expander("Add a publisher or context feed"):
        with st.form("source-profile"):
            key = st.text_input("Source ID", placeholder="independent_business")
            name = st.text_input("Source display name")
            url = st.text_input("Publisher RSS/Atom URL")
            scope = st.selectbox("Relevance scope", ["company", "sector", "market"])
            role = st.selectbox("Source role", ["independent_reporting", "independent_analysis", "primary_macro", "regulator"])
            mode = st.selectbox("Permitted content route", ["metadata_only", "feed_only", "page"])
            group = st.text_input("Original publisher group")
            documentation = st.text_input("Source documentation / access-terms URL")
            rights = st.text_area("Access and attribution notes")
            if st.form_submit_button("Save source profile"):
                try:
                    if not all(x.strip() for x in (name, url, group, documentation, rights)):
                        raise ValueError("Fill the source URL, original publisher, documentation and access notes.")
                    store.settings.save_source(key, SourceProfile(name=name, url=url, scope=scope, role=role,
                        body_mode=mode, publisher_group=group, documentation=documentation, rights_note=rights))
                    st.success("Source saved. Attach its ID to a company in Companies. Access success and relevance still need checking.")
                except ValueError as exc:
                    st.error(str(exc))
    for key, info in REGISTRY.items():
        with st.expander(info["name"]):
            st.write(info["history"])
            st.caption(info["terms"])
            if info["documentation"].startswith("https:"):
                st.link_button("Source documentation", info["documentation"])
    import shutil
    st.metric("Free local disk", f"{shutil.disk_usage(root).free / 1024**3:.1f} GB")
    if st.button("Stop background worker after current run"):
        (root / "stop-worker").write_text("stop after current run\n", encoding="utf-8")
        st.success("Worker will stop after the current run. Data and schedules are preserved.")

elif page == "Companies":
    st.title("Ticker profiles")
    st.caption("Add company names, aliases and source URLs without editing Python. New page structures may need a tested adapter.")
    st.dataframe(pd.DataFrame([{"ticker": c.ticker, "company": c.name, "exchange": c.exchange,
                               "security": c.security_type, "feeds": ", ".join(c.feeds), "CIK": c.cik} for c in companies.values()]), hide_index=True, width="stretch")
    editing = st.selectbox("Edit existing profile or add a company", ["Add new company", *companies])
    original = companies.get(editing)
    value = lambda key, default="": getattr(original, key, default) if original else default
    with st.form("company"):
        a, b = st.columns(2)
        ticker = a.text_input("Ticker symbol", value=value("ticker"), placeholder="IBM")
        name = b.text_input("Company name", value=value("name"), placeholder="International Business Machines")
        aliases = st.text_input("Aliases, comma-separated", value=", ".join(value("aliases", [])), placeholder="IBM")
        feeds = st.text_area("RSS/Atom feed URLs, one per line", value="\n".join(value("feeds", [])))
        issuer_feed = st.checkbox("These feeds belong to the company's own newsroom", value=value("feed_scope") == "company")
        cik = st.text_input("SEC CIK (optional)", value=value("cik") or "")
        source_ids = st.multiselect("Publisher/context source profiles", list(store.settings.source_profiles()), default=value("source_ids", []))
        sector_terms = st.text_area("Sector context terms, comma-separated", value=", ".join(value("sector_terms", [])))
        forms = st.text_input("SEC forms, comma-separated", value=", ".join(value("sec_forms", ["8-K", "8-K/A"])))
        st.caption("For BABA retain 6-K/20-F forms. Editing preserves its NYSE ADS identity and issuer archive configuration.")
        st.caption("Use publisher-supported feeds and permitted sources. API credentials never belong in feed URLs.")
        if st.form_submit_button("Save company", type="primary"):
            try:
                profile = original.model_dump() if original else {}
                profile.update(ticker=ticker, name=name, aliases=[x.strip() for x in aliases.split(",") if x.strip()],
                               feeds=[x.strip() for x in feeds.splitlines() if x.strip()],
                               feed_scope="company" if issuer_feed else "general", cik=cik or None,
                               source_ids=source_ids, sector_terms=[x.strip() for x in sector_terms.split(",") if x.strip()],
                               sec_forms=[x.strip() for x in forms.split(",") if x.strip()])
                store.settings.save_company(Company.model_validate(profile))
                st.success("Company profile saved. It will be available on the next page refresh.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))

elif page == "Corpus quality":
    st.title("Assess corpus quality")
    st.caption("Counts and source variety are diagnostics. They do not certify historical availability, relevance precision or forecasting value.")
    ticker = st.selectbox("Ticker", tickers)
    use_dates = st.checkbox("Filter quality report by publication dates")
    since = until = None
    if use_dates:
        a, b = st.columns(2)
        since = str(a.date_input("Quality publication from", date(2025, 1, 1)))
        until = str(b.date_input("Quality publication through", date.today()))
    report = corpus_report(store, ticker, since=since, until=until)
    a, b, c = st.columns(3)
    a.metric("Unique headline URLs", report["unique_headline_urls"])
    b.metric("Eligible body URLs", report["eligible_body_urls"])
    c.metric("Publisher groups", len(report["publisher_groups"]))
    for warning in report["warnings"]:
        st.warning(warning)
    st.write("**Headline origins and input scopes**")
    st.json({key: report[key] for key in ("publisher_groups", "source_roles", "relevance_scopes", "publication_precision", "price_recap_candidates")})
    with st.expander("Potential repeated headlines and report limits"):
        st.json({"near_duplicates": report["near_duplicate_review"], "limits": report["limits"]})
    st.download_button("Download corpus quality report", json.dumps(report, indent=2), file_name=f"{ticker}-quality.json")
    if st.button("Create frozen relevance audit"):
        spec = create_audit(store, ticker, since=since, until=until)
        st.success(f"Audit {spec['audit_id']} saved with {len(spec['records'])} items. Open Audit to review it.")

elif page == "Audit":
    st.title("Review a frozen corpus sample")
    st.caption("Random samples cover accepted and rejected headlines separately. Algorithm decisions are hidden while labelling; reviews never enter news exports.")
    audits = store.rows("SELECT id,created_at,spec FROM audits ORDER BY created_at DESC")
    if not audits:
        st.info("Create a frozen sample from Corpus quality or the audit-create command.")
    else:
        index = st.selectbox("Frozen audit", range(len(audits)), format_func=lambda i: f"{json.loads(audits[i]['spec'])['ticker']} · {audits[i]['id']}")
        spec = json.loads(audits[index]["spec"])
        reviewer = st.text_input("Audit reviewer name", value="Researcher")
        if spec["records"]:
            chosen = st.selectbox("Sample item", range(len(spec["records"])), format_func=lambda i: f"Item {i+1} · {spec['records'][i]['headline']}")
            item = spec["records"][chosen]
            st.subheader(item["headline"])
            st.link_button("Original source for verification", item["url"])
            st.write(item["body"] or "No extracted body in this frozen sample. Mark text quality Cannot assess.")
            st.json({"publication_assertion": item["published"], "first_seen": item["first_seen"]})
            if item.get("raw_hash"):
                st.download_button("Download saved source for this sample item", store.read_artifact(item["raw_hash"]),
                                   file_name=item["record_id"] + "-source.html")
            with st.form("audit-review"):
                relevant = st.radio("Relevant to the company, its configured sector or market context?", ["Relevant", "Irrelevant", "Uncertain"])
                text_quality = st.radio("Text fidelity versus the saved/original source", ["Cannot assess", "Faithful", "Incomplete/boilerplate"])
                timing = st.radio("Timestamp evidence", ["Uncertain", "Supported", "Conflicting"])
                note = st.text_area("Audit notes")
                if st.form_submit_button("Save audit review"):
                    try:
                        save_audit_review(store, spec["audit_id"], item["record_id"], reviewer, relevant, text_quality, timing, note)
                        st.success("Audit review saved separately from corpus eligibility.")
                    except ValueError as exc:
                        st.error(str(exc))
        result = audit_report(store, spec["audit_id"], reviewer)
        st.json(result)
        st.download_button("Download frozen sample", json.dumps(spec, indent=2), file_name=spec["audit_id"] + ".json")
        st.download_button("Download audit results", json.dumps(result, indent=2), file_name=spec["audit_id"] + "-review.json")

st.divider()
st.caption("Newsroom · News collection and optional headline experiments · Results save locally")
