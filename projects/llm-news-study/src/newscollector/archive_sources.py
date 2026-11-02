"""Public dated headline indexes; no body extraction or titles invented from URLs.

Source/layout evidence: SOURCE_AUDIT.md and CORPUS_RESEARCH.md (2026-11-02).
Entry points: https://technode.com/2025/10/,
https://www.alibabacloud.com/en/press-room, https://www.cnbc.com/sitemapAll.xml.
These live layouts can change; raise/report incomplete coverage rather than infer
missing publication metadata. Raw responses are archived by the shared fetcher.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from urllib.parse import urljoin, urlsplit

from dateutil.parser import parse as parse_date
from lxml import html, etree

from .core import Candidate, publication_day, timestamp


def months(since, until):
    current = until.replace(day=1)
    lower = since.replace(day=1)
    while current >= lower:
        yield current
        current = (current - timedelta(days=1)).replace(day=1)


def technode_rows(result):
    tree = html.fromstring(result.body.decode("utf-8"))
    rows = []
    for article in tree.xpath('//article[contains(concat(" ",normalize-space(@class)," ")," type-post ")]'):
        links = article.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," entry-title ")]/a')
        times = article.xpath('.//time[contains(concat(" ",normalize-space(@class)," ")," published ")]/@datetime')
        if not times:
            times = article.xpath('.//time/@datetime')[:1]
        if not links or not times:
            raise ValueError("TechNode archive changed: an article has no explicit headline/publication timestamp.")
        link = links[0]
        target = urljoin(result.url, link.get("href", ""))
        if urlsplit(target).hostname != "technode.com":
            continue
        headline = " ".join(link.text_content().split())
        published = {**timestamp(times[0]), "semantic": "publisher_archive_publication"}
        sponsored = any(x in article.get("class", "").lower() for x in ("sponsored", "branded"))
        rows.append(Candidate(url=target, title=headline, publisher="TechNode", source="archives",
            source_item_id=article.get("id", target), published=published,
            kind="sponsored" if sponsored else "reporting", discovery_artifact=result.artifact,
            metadata={"scope": "sector", "role": "unclassified_reporting" if sponsored else "independent_reporting",
                "publisher_group": "TechNode", "body_mode": "metadata_only", "index_url": result.url,
                "rights_note": "Public dated index headlines/links with attribution; no article bodies or redistribution licence.",
                "sponsored_candidate": sponsored}))
    return rows, tree


def archive_candidates(fetcher, company, request, *, emit=None):
    """Walk bounded monthly indexes, stopping when their dated rows precede the window."""
    items, artifacts, seen, issues = [], [], set(), []
    remaining = request.discovery_pages
    # Shanghai dates shortly after midnight can belong to the previous New York day.
    upper = request.until + timedelta(days=1)
    month_list = list(months(request.since, upper))
    for month in month_list:
        if not remaining:
            issues.append("archive page budget reached before all requested months were checked")
            break
        url = f"https://technode.com/{month.year}/{month.month:02d}/"
        while url:
            if not remaining:
                issues.append("archive page budget reached; older pages remain unchecked")
                break
            remaining -= 1
            try:
                result = fetcher.get(url, delay=2)
                artifacts.append(result.artifact)
                rows, tree = technode_rows(result)
                if not rows:
                    raise ValueError("No dated headline rows found; coverage is unknown.")
                for item in rows:
                    if item.url in seen:
                        continue
                    seen.add(item.url)
                    items.append(item)
                    if emit:
                        emit(item)
                next_links = tree.xpath('//a[contains(concat(" ",normalize-space(@class)," ")," next ") and contains(concat(" ",normalize-space(@class)," ")," page-numbers ")]/@href')
                if not next_links or min(publication_day(x.published) for x in rows) < str(request.since):
                    break
                next_url = urljoin(result.url, next_links[0])
                expected = f"/{month.year}/{month.month:02d}/page/"
                if urlsplit(next_url).hostname != "technode.com" or not urlsplit(next_url).path.startswith(expected) or next_url == url:
                    raise ValueError("Unexpected archive pagination link; stopped without following it.")
                url = next_url
            except InterruptedError:
                raise
            except Exception as exc:
                issues.append(f"{url}: {type(exc).__name__}: {exc}")
                break
    note = f"Checked {len(artifacts)} TechNode dated archive pages; metadata only. Publisher dates are assertions; no historical text-version certification."
    if issues:
        note += " INCOMPLETE: " + "; ".join(issues)
    if not artifacts and issues:
        raise ValueError(note)
    return items, note, artifacts


def cloud_candidates(fetcher, company, request, *, emit=None):
    if company.ticker != "BABA":
        raise ValueError("Alibaba Cloud press room belongs to the BABA profile only.")
    result = fetcher.get("https://www.alibabacloud.com/en/press-room", delay=2)
    text = result.body.decode("utf-8")
    match = re.search(r'window\.pressPageData\["tab0"\]\s*=\s*', text)
    if not match:
        raise ValueError("Alibaba Cloud press-room data changed; no supported dated announcement list.")
    data, _ = json.JSONDecoder().raw_decode(text[match.end():])
    items, seen = [], set()
    for group in data:
        for row in group.get("contentArrItem", []):
            target = row.get("readMoreLink", "")
            headline = " ".join(row.get("heading", "").split())
            if not headline or not target or target in seen:
                continue
            if urlsplit(target).hostname != "www.alibabacloud.com":
                continue
            raw = f"{row.get('date', '')} {row.get('year', '')}"
            if not row.get("date") or not re.fullmatch(r"\d{4}", str(row.get("year", ""))):
                raise ValueError("Alibaba Cloud announcement has no complete publication date.")
            day = parse_date(raw.replace("Sept ", "Sep ")).date().isoformat()
            item = Candidate(url=target, title=headline, publisher="Alibaba Cloud", source="cloud",
                source_item_id=target, published={**timestamp(day), "source_raw": raw, "semantic": "issuer_press_room_date"},
                company_scoped=True, kind="issuer_release", discovery_artifact=result.artifact,
                metadata={"scope": "company", "role": "issuer", "publisher_group": "Alibaba Group",
                    "body_mode": "metadata_only", "index_url": result.url,
                    "rights_note": "Issuer-distributed announcement headline/link metadata with attribution. Not independent reporting."})
            seen.add(target)
            items.append(item)
            if emit:
                emit(item)
    if not items:
        raise ValueError("Alibaba Cloud press room contained no supported dated announcements.")
    return items, "Dated announcements currently exposed in the public press-room index; not guaranteed complete. Date-only issuer assertions; metadata only.", [result.artifact]


def _likely_url(url, company):
    words = re.sub(r"[^a-z0-9]+", " ", urlsplit(url).path.lower())
    def present(term):
        term = re.sub(r"[^a-z0-9]+", " ", term.lower()).strip()
        return bool(term and re.search(r"\b" + re.escape(term) + r"\b", words))
    company_hit = any(present(x) for x in [company.ticker, company.name, *company.aliases])
    context_hit = any(any(present(a) for a in r.get("anchors", [])) and any(present(t) for t in r.get("topics", []))
                      for r in company.context_rules) or any(present(x) for x in company.sector_terms)
    return 2 if company_hit else 1 if context_hit else 0


def cnbc_metadata(result):
    """Read published metadata only. Sitemap lastmod and URL slugs are never headlines."""
    tree = html.fromstring(result.body.decode("utf-8"))
    titles = tree.xpath('//meta[@property="og:title"]/@content')
    dates = tree.xpath('//meta[@property="article:published_time"]/@content')
    if not titles or not dates:
        raise ValueError("Missing explicit public headline/publication metadata; no inferred headline saved.")
    headline = " ".join(titles[0].split())
    pub = timestamp(dates[0])
    if not headline or not pub.get("utc"):
        raise ValueError("Unusable article publication metadata.")
    paywalled_body = False
    for node in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            data = json.loads(node.text or "")
        except ValueError:
            continue
        objects = data if isinstance(data, list) else [data]
        for obj in objects:
            if isinstance(obj, dict) and obj.get("isAccessibleForFree") in (False, "False", "false"):
                paywalled_body = True
    descriptions = tree.xpath('//meta[@name="description"]/@content')
    return Candidate(url=result.url, title=headline, publisher="CNBC", source="news_sites",
        source_item_id=result.url, published={**pub, "semantic": "publisher_article_publication_metadata"},
        discovery_artifact=result.artifact, kind="reporting",
        metadata={"scope": "sector", "role": "independent_reporting", "publisher_group": "CNBC",
            "body_mode": "metadata_only", "description": descriptions[0] if descriptions else "",
            "body_access": "subscription_flagged" if paywalled_body else "not_requested",
            "rights_note": "Public headline/link metadata only, with attribution. No article text extraction, premium access or redistribution licence.",
            "updated_at_assertion": tree.xpath('//meta[@property="article:modified_time"]/@content')})


def news_site_candidates(fetcher, company, request, *, emit=None):
    index = fetcher.get("https://www.cnbc.com/sitemapAll.xml", delay=2)
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    tree = etree.fromstring(index.body, parser)
    maps = [u for u in tree.xpath('//*[local-name()="loc"]/text()')
            if re.fullmatch(r"https://www\.cnbc\.com/CNBCsitemapAll\d+\.xml", u)]
    if not maps:
        raise ValueError("CNBC sitemap index changed; no supported archive indexes.")
    maps.sort(key=lambda u: int(re.search(r"All(\d+)", u).group(1)), reverse=True)
    artifacts, items, issues, found = [index.artifact], [], [], {}
    # Broad indexes contain mixed dates. Scan a declared bounded subset rather than claim completeness.
    for url in maps[:min(3, request.discovery_pages)]:
        try:
            result = fetcher.get(url, delay=2)
            artifacts.append(result.artifact)
            root = etree.fromstring(result.body, etree.XMLParser(resolve_entities=False, no_network=True))
            for target in root.xpath('//*[local-name()="loc"]/text()'):
                parts = urlsplit(target)
                m = re.match(r"/(\d{4})/(\d{2})/(\d{2})/", parts.path)
                if parts.hostname != "www.cnbc.com" or not m or any(x in parts.path for x in ('/video/', '/pro/', '/investingclub/')):
                    continue
                day = date(*map(int, m.groups()))
                if request.since - timedelta(days=1) <= day <= request.until + timedelta(days=1):
                    score = _likely_url(target, company)
                    if score:
                        found[target] = score
        except InterruptedError:
            raise
        except Exception as exc:
            issues.append(f"{url}: {type(exc).__name__}: {exc}")
    cap = min(100, request.discovery_pages * 4)
    candidates = sorted(found, key=lambda u: (-found[u], u))
    if len(candidates) > cap:
        issues.append(f"public metadata request cap reached: {cap} of {len(candidates)} candidate links checked")
    for target in candidates[:cap]:
        try:
            result = fetcher.get(target, delay=2)
            artifacts.append(result.artifact)
            item = cnbc_metadata(result)
            items.append(item)
            if emit:
                emit(item)
        except InterruptedError:
            raise
        except Exception as exc:
            issues.append(f"{target}: {type(exc).__name__}: {exc}")
    note = f"INCOMPLETE: CNBC public sitemap discovery ({min(3, request.discovery_pages)} of {len(maps)} indexes), {min(cap, len(candidates))} metadata pages checked. No bodies or premium pages; publication metadata may have been revised."
    if issues:
        note += " Issues: " + "; ".join(issues)
    return items, note, artifacts
