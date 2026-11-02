"""Deterministic extraction from archived bytes with evidence for dates."""
from __future__ import annotations

import importlib.metadata
import json
import re
import io
from urllib.parse import urlsplit, urljoin

import trafilatura
from lxml import etree, html

from .core import timestamp


def linked_pdf(body: bytes, url: str) -> str | None:
    """Only a tested issuer's explicit PDF-only release; never guess or bypass access."""
    if (urlsplit(url).hostname or "").lower() != "www.alibabagroup.com":
        return None
    tree = html.fromstring(body)
    visible = " ".join(tree.text_content().split())
    if "this press release is available in pdf version only" not in visible.casefold():
        return None
    targets = list(dict.fromkeys(urljoin(url, a.get("href", "")) for a in tree.xpath('//a[@href]')
                                if urlsplit(a.get("href", "")).path.casefold().endswith(".pdf")))
    allowed = [target for target in targets if urlsplit(target).scheme == "https" and
               urlsplit(target).hostname in ("data.alibabagroup.com", "www.alibabagroup.com")]
    if len(allowed) != 1:
        raise ValueError("PDF-only release has missing/ambiguous official PDF links; headline retained, full body unavailable.")
    return allowed[0]


def extract_pdf(body: bytes) -> dict:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is not processed.")
        if len(reader.pages) > 80:
            raise ValueError("PDF exceeds the 80-page processing bound; no truncated full article is exported.")
        pages = []
        for index, page in enumerate(reader.pages):
            stream = page.get_contents()
            if stream is not None and len(stream.get_data()) > 8 * 1024 * 1024:
                raise ValueError("PDF page content stream exceeds the processing bound.")
            text = (page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or "").strip()
            if len(text) < 40:
                raise ValueError(f"PDF page {index+1} lacks a sufficient text layer; OCR/unreadable pages are not silently omitted.")
            pages.append(text)
        text = "\n\n".join(f"[Page {i+1}]\n{page}" for i, page in enumerate(pages))
        if len(text) < 80:
            raise ValueError("PDF contains no complete readable text.")
        return {"title": "", "body": text, "paragraphs": [p for page in pages for p in re.split(r"\n\s*\n", page) if p.strip()],
                "published": timestamp(None), "updated": timestamp(None), "date_assertions": [], "date_conflict": False,
                "extractor": "pypdf-layout/" + importlib.metadata.version("pypdf"),
                "pdf_pages": len(pages), "layout_note": "Text layer with layout spacing, not verified semantic table reconstruction or OCR."}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not extract PDF text: {type(exc).__name__}: {exc}") from exc


def extract(body: bytes, url: str, content_type: str = "") -> dict:
    if "pdf" in content_type.lower() or body.startswith(b"%PDF"):
        return extract_pdf(body)
    try:
        tree = html.fromstring(body)
    except (etree.ParserError, ValueError) as exc:
        raise ValueError("Could not parse the saved source as an HTML document.") from exc
    meta: dict[str, str] = {}
    if linked_pdf(body, url):
        raise ValueError("PDF-only release landing page is not an article body; collect the linked official PDF.")
    for tag in tree.xpath("//meta[@content]"):
        meta[(tag.get("property") or tag.get("name") or "").lower()] = tag.get("content", "")
    title = meta.get("og:title") or meta.get("twitter:title") or " ".join(tree.xpath("//h1[1]//text()")) or " ".join(tree.xpath("//title/text()"))
    published = timestamp(meta.get("article:published_time") or meta.get("date") or meta.get("datepublished"))
    updated = timestamp(meta.get("article:modified_time") or meta.get("datemodified"))
    structured = []
    for node in tree.xpath('//script[@type="application/ld+json"]/text()'):
        try:
            data = json.loads(node)
            todo = data if isinstance(data, list) else [data]
            while todo:
                item = todo.pop()
                if not isinstance(item, dict):
                    continue
                if item.get("isAccessibleForFree") is False or item.get("isAccessibleForFree") == "false":
                    raise ValueError("Source declares the article paywalled; no bypass attempted.")
                if isinstance(item.get("@graph"), list):
                    todo.extend(item["@graph"])
                if item.get("datePublished"):
                    structured.append(timestamp(item["datePublished"]))
                if not updated["raw"] and item.get("dateModified"):
                    updated = timestamp(item["dateModified"])
                if not title and item.get("headline"):
                    title = item["headline"]
        except (json.JSONDecodeError, TypeError):
            continue
    if not published["raw"] and structured:
        published = structured[0]
    # Company newsroom/filing containers preserve short releases and tabular facts.
    selectors = [
        '//div[contains(concat(" ",normalize-space(@class)," ")," article-body ")]',
        '//div[contains(@class,"article__body")]',
        '//div[contains(@class,"body-copy")]',
    ]
    chosen = next((nodes[0] for selector in selectors if (nodes := tree.xpath(selector))), None)
    method = "trafilatura/" + importlib.metadata.version("trafilatura")
    text = ""
    if chosen is not None:
        paragraphs = [re.sub(r"\s+", " ", p.text_content()).strip() for p in chosen.xpath('.//p|.//h2|.//h3|.//li|.//tr')]
        text = "\n\n".join(p for p in paragraphs if p)
        method = "newsroom-structural/1"
    if len(text) < 100:
        text = trafilatura.extract(body, url=url, include_comments=False, include_tables=True,
                                   favor_precision=False, deduplicate=False) or ""
        method = "trafilatura/" + importlib.metadata.version("trafilatura")
    text = text.strip()
    host = (urlsplit(url).hostname or "").lower()
    if host in ("apple.com", "www.apple.com") and "/newsroom/" in url:
        # Known Apple newsroom sharing/media controls are not article paragraphs.
        text = re.sub(r"\nShare article\s*\nMedia\s*\n-?\s*Text of this article.*$", "", text, flags=re.S).strip()
    if host == "blogs.microsoft.com":
        text = re.sub(r"\nThe post [^\n]+ appeared first on The Official Microsoft Blog\.\s*$", "", text).strip()
    if host == "www.alibabagroup.com":
        title = re.sub(r"\s*-\s*Alibaba Group$", "", title).strip()
        text = re.sub(r"\nPDF[：:]\s*$", "", text).strip()
    if len(text) < 80:
        raise ValueError("No complete readable article body found (fewer than 80 characters).")
    if re.search(r"^(access denied|just a moment|verify you are human|enable javascript)", text[:100], re.I):
        raise ValueError("Source returned an access/verification page instead of an article.")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    assertions = [published, *structured]
    instants = {x["utc"] for x in assertions if x.get("utc")}
    return {"title": re.sub(r"\s+", " ", title).strip(), "body": text,
            "paragraphs": paragraphs, "published": published, "updated": updated,
            "extractor": method, "date_assertions": assertions,
            "date_conflict": len(instants) > 1}
