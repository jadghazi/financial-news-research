"""Bounded feed, SEC and GDELT adapters. Every limitation is returned as coverage."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlsplit, urljoin, quote_plus

import feedparser
from dateutil.parser import parse as parse_date
from lxml import html

from .core import Candidate, Company, Request, SourceProfile, timestamp
from .fetch import Fetcher

REGISTRY = {
    "news_sites": {"name": "CNBC · public news archive", "key": False,
                   "history": "Bounded public sitemap discovery, original headline/publication metadata; no complete-history guarantee.",
                   "terms": "Public headline/date/link metadata only; no bodies or access-control bypass. Subscription body flags retained; Pro/Investing Club excluded.",
                   "documentation": "https://www.cnbc.com/terms/"},
    "archives": {"name": "TechNode · dated news archive", "key": False,
                 "history": "Public monthly China-technology news indexes with explicit dates; bounded pagination.",
                 "terms": "Headline/link metadata only with publisher credit; no full article retrieval or blanket redistribution licence.",
                 "documentation": "https://technode.com/about/"},
    "cloud": {"name": "Alibaba Cloud · dated announcements", "key": False,
              "history": "Dated public press-room announcements, including 2025; issuer-originated, not independent reporting.",
              "terms": "Announcement headline/link metadata only, with issuer credit.",
              "documentation": "https://www.alibabacloud.com/en/press-room"},
    "social": {"name": "Public social posts · reviewed Bluesky accounts (optional)", "key": False,
               "history": "Pinned account identities; at most five recent pages/account. Not a historical X scraper.",
               "terms": "Public metadata only. Social posts retain their own type and timestamps; no engagement counts in exports.",
               "documentation": "https://docs.bsky.app/docs/api/app-bsky-feed-get-author-feed"},
    "aggregators": {"name": "News aggregator · multiple publishers", "key": False,
                    "history": "Public Stock Analysis recent-news cards; limited history, no paid archive.",
                    "terms": "Robots-aware headline/link collection with attribution; linked publishers retain rights.",
                    "documentation": "https://stockanalysis.com/terms-of-use/"},
    "regulatory": {"name": "Trade & technology policy · Federal Register", "key": False,
                   "history": "Public, date-searchable regulatory metadata; bounded pagination. Not financial reporting.",
                   "terms": "Government metadata; informational web rendition, official edition on govinfo.gov.",
                   "documentation": "https://www.federalregister.gov/developers/documentation/api/v1"},
    "sec_news": {"name": "SEC news · press & enforcement releases", "key": False,
                 "history": "Recent SEC-authored RSS items, not issuer filings; real contact email required.",
                 "terms": "SEC fair-access policy; no account or API key needed.",
                 "documentation": "https://www.sec.gov/about/rss-feeds"},
    "feeds": {"name": "Company newsroom RSS / Atom", "key": False,
              "history": "Only items retained in each feed; older requested dates can be missing.",
              "terms": "Check each publisher's terms; local evidence is not a redistribution license.",
              "documentation": "Configured publisher feed and robots.txt"},
    "sec": {"name": "SEC EDGAR issuer disclosures", "key": False,
            "history": "Submissions plus referenced historical files; bounded by per-job limits.",
            "terms": "NEWS_COLLECTOR_CONTACT is required; issuer exhibits retain their own provenance.",
            "documentation": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"},
    "gdelt": {"name": "GDELT DOC discovery", "key": False,
              "history": "Optional broad discovery; recent local probes timed out. Historical availability uncertain.",
              "terms": "Links only; publisher robots/access rules still apply to full-text fetching.",
              "documentation": "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/"},
    "issuer": {"name": "Issuer investor-news archive (BABA)", "key": False,
               "history": "Dated links exposed by the issuer index; first-page bounds are recorded, not a complete archive.",
               "terms": "Issuer-originated material, preserved with provenance. Current pages do not prove historical versions.",
               "documentation": "https://www.alibabagroup.com/en-US/ir-news-filings"},
    "publishers": {"name": "Independent publisher feeds", "key": False,
                   "history": "Configured publisher feeds; source-specific historical search where tested. Coverage can be sparse.",
                   "terms": "Each profile records access mode, origin, attribution and rights notes. Headlines and bodies have different availability.",
                   "documentation": "Configured source profiles"},
    "context": {"name": "Sector and market context feeds", "key": False,
                "history": "Separate context scope. Retained feeds do not constitute complete historical macro coverage.",
                "terms": "Topic matching is not evidence of a price effect; inspect the source and time assertions.",
                "documentation": "Configured source profiles"},
}


def _feed_items(result, *, source: str, company_scoped: bool = False, profile: SourceProfile | None = None,
                profile_id: str | None = None) -> list[Candidate]:
    feed = feedparser.parse(result.body)
    if not feed.entries:
        return []
    items = []
    for entry in feed.entries:
        mode = profile.body_mode if profile else "page"
        embedded = next((c.get("value", "") for c in entry.get("content", [])
                         if mode != "metadata_only" and c.get("type", "text/html") in ("text/html", "application/xhtml+xml")
                         and len(c.get("value", "")) >= 800), None)
        meta = {"feed_url": result.url, "feed_fetched_at": result.at, "embedded_html": embedded,
                "description": " ".join(html.fromstring("<div>" + entry.get("summary", "") + "</div>").text_content().split())[:2000],
                "feed_parse_warning": bool(feed.bozo), "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "feed_rights": entry.get("rights") or feed.feed.get("rights"), "body_mode": mode,
                "scope": profile.scope if profile else "company", "profile_id": profile_id,
                "role": profile.role if profile else "issuer" if company_scoped else "unclassified_reporting"}
        if profile:
            meta.update(source_profile=profile.model_dump(), publisher_group=profile.publisher_group,
                        license_url=profile.license_url, rights_note=profile.rights_note)
        items.append(Candidate(url=entry.get("link", ""), title=entry.get("title", "Untitled feed item"),
                               publisher=feed.feed.get("title", urlsplit(result.url).hostname or "Unknown"),
                               source=source, source_item_id=str(entry.get("id", entry.get("link", ""))),
                               published=timestamp(entry.get("published") or entry.get("updated")),
                               company_scoped=company_scoped,
                               kind="issuer_release" if company_scoped else "analysis" if profile and profile.role == "independent_analysis" else "macro_release" if profile and profile.role == "primary_macro" else "reporting",
                               discovery_artifact=result.artifact, metadata=meta))
    return items


def feed_candidates(fetcher: Fetcher, company: Company, request: Request, *, emit=None) -> tuple[list[Candidate], str, list[str]]:
    items, artifacts, issues = [], [], []
    if not company.feeds:
        raise ValueError("No RSS/Atom feed configured for this ticker. Add a feed in Companies or choose SEC/GDELT.")
    for feed_url in company.feeds:
        try:
            result = fetcher.get(feed_url)
            artifacts.append(result.artifact)
            feed = feedparser.parse(result.body)
            if not feed.entries:
                raise ValueError("No feed entries found; source may have changed format.")
            for entry in feed.entries:
                url = entry.get("link", "")
                pub = timestamp(entry.get("published") or entry.get("updated"))
                embedded = next((c.get("value", "") for c in entry.get("content", [])
                                 if c.get("type", "text/html") in ("text/html", "application/xhtml+xml")
                                 and len(c.get("value", "")) >= 800), None)
                # Retain out-of-window items too; the worker records a reason for each.
                items.append(Candidate(url=url, title=entry.get("title", "Untitled feed item"),
                                       publisher=feed.feed.get("title", urlsplit(feed_url).hostname or "Unknown"),
                                       source="feeds", source_item_id=str(entry.get("id", url)), published=pub,
                                       company_scoped=company.feed_scope == "company", kind="issuer_release" if company.feed_scope == "company" else "reporting", discovery_artifact=result.artifact,
                                       metadata={"feed_url": feed_url, "feed_fetched_at": result.at,
                                                 "embedded_html": embedded,
                                                 "feed_parse_warning": bool(feed.bozo)}))
                if emit:
                    emit(items[-1])
        except (ValueError, RuntimeError) as exc:
            issues.append(f"{feed_url}: {exc}")
    if not artifacts:
        raise ValueError("; ".join(issues))
    note = "Recent feed only; requested historical coverage is not guaranteed."
    if issues:
        note += " Feed failures: " + "; ".join(issues)
    return items, note, artifacts


def sec_candidates(fetcher: Fetcher, company: Company, request: Request, *, emit=None) -> tuple[list[Candidate], str, list[str]]:
    if not fetcher.store.settings.contact or "@" not in fetcher.store.settings.contact:
        raise ValueError("SEC requires a contact User-Agent. Set NEWS_COLLECTOR_CONTACT to your real contact email.")
    if not company.cik:
        raise ValueError("No CIK configured for this ticker; add its SEC CIK in Companies.")
    cik = str(int(company.cik)).zfill(10)
    fetched = fetcher.get(f"https://data.sec.gov/submissions/CIK{cik}.json", delay=1)
    data = json.loads(fetched.body)
    chunks = [(data.get("filings", {}).get("recent", {}), fetched.artifact, fetched.at)]
    artifacts = [fetched.artifact]
    if emit:
        for candidate in _sec_items(chunks, company, cik):
            emit(candidate)
    files = data.get("filings", {}).get("files", [])
    for file in files:
        if file.get("filingTo", "9999") < str(request.since) or file.get("filingFrom", "0000") > str(request.until):
            continue
        if len(artifacts) >= 12:
            return _sec_items(chunks, company, cik), "INCOMPLETE: historical submission-file request budget reached.", artifacts
        name = file.get("name", "")
        if "/" in name or not name.endswith(".json"):
            raise ValueError("Invalid historical SEC file reference.")
        historical = fetcher.get("https://data.sec.gov/submissions/" + name)
        artifacts.append(historical.artifact)
        chunks.append((json.loads(historical.body), historical.artifact, historical.at))
        if emit:
            for candidate in _sec_items([chunks[-1]], company, cik):
                emit(candidate)
    return _sec_items(chunks, company, cik), "Bounded " + ", ".join(company.sec_forms) + " discovery; primary documents only, not linked exhibits.", artifacts


def _sec_items(chunks, company: Company, cik: str) -> list[Candidate]:
    items = []
    for data, artifact, seen in chunks:
        for i, form in enumerate(data.get("form", [])):
            if form not in company.sec_forms:
                continue
            def value(key, default=""):
                values = data.get(key, [])
                return values[i] if i < len(values) else default
            accession = value("accessionNumber")
            doc = value("primaryDocument")
            if not accession or not doc or ".." in doc:
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"
            items.append(Candidate(url=url, title=f"{company.name}: {form} filing (items {value('items') or 'unspecified'})",
                                   publisher="SEC EDGAR / issuer filing", source="sec", source_item_id=accession,
                                   published=timestamp(value("filingDate")),
                                   provider_time={**timestamp(value("acceptanceDateTime")), "semantic": "filing_acceptance"},
                                   company_scoped=True, kind="filing", discovery_artifact=artifact,
                                   metadata={"cik": cik, "accession": accession, "items": value("items"),
                                             "acceptance_raw": value("acceptanceDateTime"), "metadata_fetched_at": seen,
                                             "exhibits": "Primary filing only; linked exhibit bodies are not automatically collected."}))
    return items


def gdelt_candidates(fetcher: Fetcher, company: Company, request: Request, *, emit=None) -> tuple[list[Candidate], str, list[str]]:
    begin = datetime.combine(request.since, datetime.min.time(), UTC)
    end = datetime.combine(request.until + timedelta(days=1), datetime.min.time(), UTC)
    windows, artifacts, items = [(begin, end)], [], []
    truncated = False
    while windows and len(artifacts) < 12:
        start, stop = windows.pop(0)
        terms = company.discovery_terms or [company.name]
        terms = [re.sub(r'[^\w .\-]', ' ', x).strip()[:70] for x in terms[:6]]
        query = "(" + " OR ".join('"' + x + '"' for x in terms if x) + ") sourcelang:english"
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 250,
                  "sort": "DateAsc", "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                  "enddatetime": stop.strftime("%Y%m%d%H%M%S")}
        fetched = fetcher.get("https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params), delay=6)
        artifacts.append(fetched.artifact)
        try:
            articles = json.loads(fetched.body).get("articles", [])
        except json.JSONDecodeError as exc:
            raise ValueError("GDELT returned a non-JSON response: " + fetched.body[:160].decode("utf-8", "replace")) from exc
        if len(articles) >= 250:
            if stop - start > timedelta(minutes=30):
                middle = start + (stop - start) / 2
                windows.extend([(start, middle), (middle - timedelta(seconds=1), stop)])
            else:
                truncated = True
        for article in articles:
            seen = article.get("seendate", "")
            parsed = {}
            try:
                parsed = {"raw": seen, "utc": datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat(),
                          "precision": "provider_batch", "semantic": "index_seen"}
            except ValueError:
                parsed = {"raw": seen, "utc": None, "semantic": "index_seen"}
            items.append(Candidate(url=article.get("url", ""), title=article.get("title", "Untitled"),
                                   publisher=article.get("domain", "Unknown"), source="gdelt",
                                   provider_time=parsed, discovery_artifact=fetched.artifact,
                                   metadata={"language": article.get("language"), "query": query, "role": "unclassified_reporting",
                                             "scope": "company", "publisher_group": article.get("domain", "Unknown")}))
            if emit:
                emit(items[-1])
    note = "GDELT discovery is not exhaustive; provider indexing time is not publication time."
    if truncated or windows:
        note += " INCOMPLETE: result/window request cap reached."
    return items, note, artifacts


def issuer_candidates(fetcher: Fetcher, company: Company, request: Request, *, emit=None):
    if not company.issuer_index:
        raise ValueError("No supported investor-news archive configured for this company.")
    if company.issuer_index != "https://www.alibabagroup.com/en-US/ir-news-filings":
        raise ValueError("This archive layout needs a tested source adapter; it is not a generic HTML scraper.")
    result = fetcher.get(company.issuer_index, delay=2)
    tree = html.fromstring(result.body)
    items, seen, dates = [], set(), []
    for node in tree.xpath('//a[contains(@href,"/en-US/document-")]'):
        url = urljoin(result.url, node.get("href", ""))
        if url in seen:
            continue
        titles = node.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," component-title ")]')
        time_nodes = node.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," date-date ")]')
        if not titles or not time_nodes:
            raise ValueError("Issuer index layout changed: title/date evidence is missing.")
        title = " ".join(titles[0].text_content().split())
        raw_date = " ".join(time_nodes[0].text_content().split())
        day = parse_date(raw_date).date().isoformat()
        published = {**timestamp(day), "source_raw": raw_date, "semantic": "issuer_index_date"}
        item = Candidate(url=url, title=title, publisher="Alibaba Group investor relations", source="issuer",
                         source_item_id=url.rsplit("/", 1)[-1], published=published, company_scoped=True,
                         kind="issuer_release", discovery_artifact=result.artifact,
                         metadata={"index_url": result.url, "role": "issuer", "scope": "company",
                                   "publisher_group": "Alibaba Group", "index_fetched_at": result.at})
        seen.add(url)
        dates.append(day)
        items.append(item)
        if emit:
            emit(item)
    if not items:
        raise ValueError("Issuer index returned no supported article links; coverage unknown.")
    note = f"INCOMPLETE archive: first index page only; {len(items)} links dated {min(dates)} through {max(dates)}. No historical text-version proof."
    return items, note, [result.artifact]


def _profile_candidates(fetcher: Fetcher, company: Company, request: Request, *, context: bool, emit=None):
    frozen = json.loads(fetcher.store.job(fetcher.job)["request"]).get("source_profiles")
    profiles = ({k: SourceProfile.model_validate(v) for k, v in frozen.items()} if frozen is not None
                else fetcher.store.settings.source_profiles())
    items, artifacts, issues = [], [], []
    capped = False
    selected = [(key, profiles[key]) for key in company.source_ids if key in profiles and profiles[key].enabled
                and ((profiles[key].scope == "market") == context)]
    if not selected:
        raise ValueError("No matching enabled source profiles. Configure source profiles in Sources and attach them in Companies.")
    for key, profile in selected:
        historical = bool(profile.historical_url and request.since.year < datetime.now(UTC).year)
        urls = [profile.url]
        if historical:
            urls = []
            for term in (profile.search_terms or [(company.discovery_terms or [company.name])[0]]):
                for year in range(request.since.year, min(request.until.year, request.since.year + 2) + 1):
                    for page in range(1, profile.max_pages + 1):
                        urls.append(profile.historical_url.format(query=quote_plus(term), year=year, page=page))
        seen_urls, empty_years = set(), set()
        for url in urls:
            # A short page terminates that year's search; other years remain independent.
            year = re.search(r"[?&]year=(\d{4})", url)
            query = re.search(r"[?&]s=([^&]+)", url)
            year_key = (query[1] if query else "", year[1] if year else "")
            if year and year_key in empty_years:
                continue
            try:
                result = fetcher.get(url, delay=2)
                artifacts.append(result.artifact)
                returned = _feed_items(result, source="context" if context else "publishers", profile=profile, profile_id=key)
                for candidate in returned:
                    if candidate.url in seen_urls:
                        continue
                    seen_urls.add(candidate.url)
                    items.append(candidate)
                    if emit:
                        emit(candidate)
                if historical and len(returned) < 10 and year:
                    empty_years.add(year_key)
                if historical and len(returned) >= 10 and f"paged={profile.max_pages}" in url:
                    capped = True
                if not returned:
                    issues.append(f"{profile.name}: feed returned no entries; this is not evidence of no news.")
            except (ValueError, RuntimeError) as exc:
                issues.append(f"{profile.name}: {exc}")
    note = "Bounded source-profile feeds/search; historical completeness and article revisions are unverified."
    if request.until.year - request.since.year > 2:
        note += " INCOMPLETE: historical search limited to three years."
    if capped:
        note += " INCOMPLETE: a full final search page reached the source page limit."
    if any(p.historical_url for _, p in selected):
        note += " Historical search is bounded by per-source page limits; result caps may miss items."
    if issues:
        note += " Feed failures: " + "; ".join(issues)
    if not artifacts:
        raise ValueError(note)
    return items, note, artifacts


def publisher_candidates(fetcher, company, request, *, emit=None):
    return _profile_candidates(fetcher, company, request, context=False, emit=emit)


def context_candidates(fetcher, company, request, *, emit=None):
    return _profile_candidates(fetcher, company, request, context=True, emit=emit)


from .broad_sources import aggregator_candidates, regulatory_candidates, sec_news_candidates, social_candidates
from .archive_sources import archive_candidates, cloud_candidates, news_site_candidates

ADAPTERS = {"feeds": feed_candidates, "sec": sec_candidates, "gdelt": gdelt_candidates,
            "issuer": issuer_candidates, "publishers": publisher_candidates, "context": context_candidates,
            "aggregators": aggregator_candidates, "regulatory": regulatory_candidates, "sec_news": sec_news_candidates,
            "social": social_candidates, "archives": archive_candidates, "cloud": cloud_candidates,
            "news_sites": news_site_candidates}
