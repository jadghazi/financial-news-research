"""Account-free headline discovery. Metadata does not imply a complete archive."""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode, urlsplit

from lxml import html

from .core import Candidate, SourceProfile, timestamp


def aggregator_candidates(fetcher, company, request, *, emit=None):
    # Only the public news section is parsed. No quotes, prices or scores exported.
    url = f"https://stockanalysis.com/stocks/{company.ticker.lower()}/"
    result = fetcher.get(url, delay=3)
    tree = html.fromstring(result.body)
    items, seen = [], set()
    for link in tree.xpath('//h3/a[starts-with(@href,"https://")]'):
        target = link.get("href", "")
        title = " ".join(link.text_content().split())
        parent = link.getparent().getparent()
        dates = parent.xpath('.//div[@title]')
        if not dates or parent.tag in ("html", "body"):
            continue
        raw = dates[0].get("title", "")
        if not re.match(r"\d{4}-\d{2}-\d{2}T", raw):
            continue
        if not title or target in seen:
            continue
        origin = " ".join(dates[0].text_content().split()).split(" - ", 1)
        publisher = origin[1] if len(origin) == 2 else urlsplit(target).hostname or "Unknown"
        item = Candidate(url=target, title=title, publisher=publisher, source="aggregators",
                         source_item_id=target, published={**timestamp(raw), "semantic": "aggregator_reported_publication"},
                         provider_time={**timestamp(raw), "semantic": "provider_reported_publication_not_receipt"},
                         discovery_artifact=result.artifact,
                         metadata={"scope": "sector", "role": "unclassified_reporting", "body_mode": "metadata_only",
                                   "publisher_group": publisher, "discovery_url": result.url,
                                   "aggregator": "Stock Analysis", "original_url": target,
                                   "rights_note": "Public headline/link metadata with source credit. Not a license to retrieve or redistribute linked publisher articles.",
                                   "timestamp_note": "Timestamp asserted by aggregator; publisher publication time not independently verified."})
        seen.add(target); items.append(item)
        if emit:
            emit(item)
    if not items:
        raise ValueError("No dated news cards found on the public aggregator page; layout or coverage may have changed.")
    return items, "INCOMPLETE: public recent-news cards only; no paid archive or pagination accessed. Aggregator-reported dates are unverified publisher assertions.", [result.artifact]


def regulatory_candidates(fetcher, company, request, *, emit=None):
    queries = company.regulatory_queries
    if not queries:
        raise ValueError("No regulatory search configured for this company. Add explicit regulatory_queries to its profile.")
    items, seen, artifacts, caps = [], set(), [], []
    for term in queries[:3]:
        for page in range(1, request.discovery_pages + 1):
            params = {"conditions[term]": term, "conditions[publication_date][gte]": str(request.since),
                      "conditions[publication_date][lte]": str(request.until), "per_page": 100, "page": page, "order": "oldest"}
            if company.regulatory_agencies:
                params["conditions[agencies][]"] = company.regulatory_agencies
            result = fetcher.get("https://www.federalregister.gov/api/v1/documents.json?" + urlencode(params, doseq=True), delay=2)
            artifacts.append(result.artifact)
            data = json.loads(result.body)
            if not isinstance(data.get("results"), list):
                raise ValueError("Regulatory API returned an unexpected schema; already-discovered items are preserved.")
            for row in data["results"]:
                target = row.get("html_url", "")
                if not target or target in seen:
                    continue
                agencies = ", ".join(a.get("name", "") for a in row.get("agencies", [])) or "Federal Register"
                item = Candidate(url=target, title=row.get("title", ""), publisher=agencies,
                    source="regulatory", source_item_id=row.get("document_number", ""), kind="regulatory_notice",
                    published={**timestamp(row.get("publication_date")), "semantic": "register_publication_date"},
                    discovery_artifact=result.artifact,
                    metadata={"scope": "sector", "role": "regulator", "publisher_group": "US Federal Register / " + agencies,
                              "body_mode": "metadata_only", "query": term,
                              "description": row.get("abstract"), "document_type": row.get("type"),
                              "configured_agencies": company.regulatory_agencies,
                              "rights_note": "US government regulatory title/metadata. Not independent financial reporting. The web rendition is informational; official edition is on govinfo.gov."})
                seen.add(target); items.append(item)
                if emit:
                    emit(item)
            total_pages = data.get("total_pages")
            more = bool(data.get("next_page_url")) or (isinstance(total_pages, int) and page < total_pages)
            if not more:
                break
            if page == request.discovery_pages:
                caps.append(term)
    note = "Regulatory context only; selected public searches, not an all-news archive. Date-only publication assertions do not establish intraday availability."
    if caps:
        note += " INCOMPLETE: discovery page limit reached for " + ", ".join(caps)
    return items, note, artifacts


def social_candidates(fetcher, company, request, *, emit=None):
    if not company.social_accounts:
        raise ValueError("No reviewed public social accounts are configured for this ticker.")
    items, seen, artifacts, issues = [], set(), [], []
    for account in company.social_accounts[:5]:
        handle, did = account.get("handle", ""), account.get("did", "")
        if not re.fullmatch(r"[a-z0-9.\-]+", handle) or not did.startswith("did:"):
            raise ValueError("Social accounts need a reviewed handle and a pinned DID identity.")
        cursor = None
        for page in range(min(request.discovery_pages, 5)):
            params = {"actor": did, "limit": 100, "filter": "posts_no_replies"}
            if cursor:
                params["cursor"] = cursor
            result = fetcher.get("https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?" + urlencode(params), delay=2)
            artifacts.append(result.artifact)
            data = json.loads(result.body)
            if not isinstance(data.get("feed"), list):
                raise ValueError("Public social feed returned an unexpected schema.")
            oldest = None
            for entry in data["feed"]:
                post = entry.get("post", {})
                author = post.get("author", {})
                # A trusted account reposting another account does not verify that author.
                if entry.get("reason") or author.get("did") != did or author.get("handle") != handle:
                    continue
                record = post.get("record", {})
                created = record.get("createdAt", "")
                oldest = min(oldest or created, created) if created else oldest
                uri = post.get("uri", "")
                if uri in seen or not uri.startswith("at://" + did + "/app.bsky.feed.post/"):
                    continue
                embed = post.get("embed", {}).get("external", {})
                title = (embed.get("title") or record.get("text") or "").strip()
                target = f"https://bsky.app/profile/{did}/post/{uri.rsplit('/', 1)[-1]}"
                item = Candidate(url=target, title=title, publisher=account.get("name", handle), source="social",
                    source_item_id=uri, kind="social_post", discovery_artifact=result.artifact,
                    published={**timestamp(created), "semantic": "social_post_created_not_linked_article_publication"},
                    provider_time={**timestamp(post.get("indexedAt")), "semantic": "social_indexed_at"},
                    metadata={"scope": "sector", "role": "social_account", "publisher_group": account.get("name", handle),
                              "body_mode": "metadata_only", "social_account": {"handle": handle, "did": did,
                                  "identity_evidence": account.get("identity_evidence", ""), "post_uri": uri,
                                  "linked_url": embed.get("uri"), "text_representation": "external_card_title" if embed.get("title") else "post_text"},
                              "rights_note": "Public social post/card title; distinct from a publisher article. No engagement metrics exported."})
                seen.add(uri); items.append(item)
                if emit:
                    emit(item)
            cursor = data.get("cursor")
            if not cursor or (oldest and oldest[:10] < str(request.since)):
                break
            if page == min(request.discovery_pages, 5) - 1:
                issues.append(handle + " reached the public-feed page budget")
    note = "INCOMPLETE: bounded public Bluesky account feeds, not an X archive. Post creation is not linked-article publication. Replies/reposts and engagement counts are excluded."
    if issues:
        note += " " + "; ".join(issues)
    return items, note, artifacts


def sec_news_candidates(fetcher, company, request, *, emit=None):
    from .sources import _feed_items
    if not fetcher.store.settings.contact:
        raise ValueError("SEC news requires your real contact email. Save it in Settings → Sources & connections; no account or API key needed.")
    items, artifacts, errors = [], [], []
    for label, url in [("SEC press releases", "https://www.sec.gov/news/pressreleases.rss"),
                       ("SEC litigation releases", "https://www.sec.gov/enforcement-litigation/litigation-releases/rss")]:
        try:
            result = fetcher.get(url, delay=2)
            artifacts.append(result.artifact)
            profile = SourceProfile(name=label, url=url, scope="sector", role="regulator", body_mode="metadata_only",
                                    publisher_group="US SEC", documentation="https://www.sec.gov/about/rss-feeds",
                                    rights_note="SEC-authored news; separate from issuer-originated EDGAR filings.")
            for item in _feed_items(result, source="sec_news", profile=profile):
                items.append(item)
                if emit:
                    emit(item)
        except (ValueError, RuntimeError) as exc:
            errors.append(label + ": " + str(exc))
    if not artifacts:
        raise ValueError("; ".join(errors))
    note = "SEC-authored recent feed metadata, not a complete archive or issuer filing feed."
    if errors:
        note += " Feed failures: " + "; ".join(errors)
    return items, note, artifacts
