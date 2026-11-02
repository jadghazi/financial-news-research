"""Validated configuration and timestamp semantics shared by every interface."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from dateutil.parser import parse as parse_date
from pydantic import BaseModel, Field, field_validator


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def utc(value: str | datetime) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("Use a timezone-aware timestamp, for example 2026-10-24T12:00:00Z.")
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def timestamp(value: str | None) -> dict:
    raw = str(value or "").strip()
    result = {"raw": raw or None, "utc": None, "precision": "unknown"}
    if not raw:
        return result
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z?", raw):
        return {**result, "precision": "date"}
    try:
        dt = parse_date(raw)
        if dt.tzinfo is not None:
            return {**result, "utc": utc(dt), "precision": "instant"}
        return {**result, "precision": "timezone_missing"}
    except (ValueError, OverflowError):
        return result


def digest(value: bytes | str | dict) -> str:
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def publication_day(publication: dict) -> str:
    """US study date; date-only source assertions remain date-only."""
    instant = publication.get("utc")
    if instant:
        return datetime.fromisoformat(instant.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    raw = str(publication.get("raw") or "")
    return raw[:10] if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|Z$)", raw) else ""


class Company(BaseModel):
    ticker: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    feeds: list[str] = Field(default_factory=list)
    feed_scope: Literal["company", "general"] = "general"
    cik: str | None = None
    calendar: str = "XNYS"
    exchange: str = "NYSE/Nasdaq (US)"
    security_type: str = "common stock"
    related_symbols: list[str] = Field(default_factory=list)
    sec_forms: list[str] = Field(default_factory=lambda: ["8-K", "8-K/A"])
    discovery_terms: list[str] = Field(default_factory=list)
    sector_terms: list[str] = Field(default_factory=list)
    context_rules: list[dict] = Field(default_factory=list)
    regulatory_queries: list[str] = Field(default_factory=list)
    regulatory_agencies: list[str] = Field(default_factory=list)
    social_accounts: list[dict] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    issuer_index: str | None = None



    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        value = value.upper().strip()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^]{0,14}", value):
            raise ValueError("Enter a ticker such as AAPL, MSFT or BRK.B.")
        return value

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if len(value.strip()) < 2:
            raise ValueError("A company name is required to avoid ambiguous ticker-only matches.")
        return value.strip()


class Request(BaseModel):
    tickers: list[str]
    sources: list[Literal["feeds", "sec", "gdelt", "issuer", "publishers", "context", "regulatory", "sec_news", "aggregators", "social", "archives", "cloud", "news_sites"]] = ["feeds"]
    since: date
    until: date
    max_articles: int = Field(default=20, ge=1, le=500)
    collect_bodies: bool = True
    discovery_pages: int = Field(default=10, ge=1, le=25)
    mode: Literal["once", "watch"] = "once"
    interval_minutes: int = Field(default=15, ge=5, le=1440)

    @field_validator("tickers")
    @classmethod
    def normalize(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Select at least one ticker.")
        return list(dict.fromkeys(x.strip().upper() for x in value))

    def checked(self) -> "Request":
        if self.until < self.since:
            raise ValueError("The end date must not precede the start date.")
        if not self.sources:
            raise ValueError("Select at least one source.")
        return self


class Candidate(BaseModel):
    url: str
    title: str
    publisher: str
    source: str
    source_item_id: str = ""
    published: dict = Field(default_factory=dict)
    provider_time: dict = Field(default_factory=dict)
    company_scoped: bool = False
    kind: str = "reporting"
    metadata: dict = Field(default_factory=dict)
    discovery_artifact: str | None = None


class SourceProfile(BaseModel):
    """A declared feed role is distinct from evidence that an item concerns a company."""
    name: str
    url: str
    scope: Literal["company", "sector", "market"] = "company"
    role: Literal["independent_reporting", "independent_analysis", "primary_macro", "regulator", "issuer"]
    body_mode: Literal["page", "feed_only", "metadata_only"] = "page"
    publisher_group: str
    documentation: str
    rights_note: str
    license_url: str = ""
    historical_url: str | None = None
    max_pages: int = Field(default=3, ge=1, le=5)
    search_terms: list[str] = Field(default_factory=list, max_length=6)
    enabled: bool = True


class Settings:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("NEWS_COLLECTOR_HOME", "data")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "collector.sqlite3"
        preferences = self.root / "preferences.json"
        saved = json.loads(preferences.read_text(encoding="utf-8")) if preferences.exists() else {}
        self.contact = os.environ.get("NEWS_COLLECTOR_CONTACT", saved.get("contact", "")).strip()
        from . import __version__
        self.user_agent = "NewsStudyCollector/" + __version__ + (f" (contact: {self.contact})" if self.contact else " (local research)")
        self.max_bytes = 8 * 1024 * 1024
        self.min_free_bytes = 200 * 1024 * 1024

    def save_contact(self, value: str) -> None:
        value = value.strip()
        if value and ("@" not in value or any(c in value for c in "\r\n")):
            raise ValueError("Enter a real contact email, or leave it empty to disable SEC requests.")
        path = self.root / "preferences.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"contact": value}), encoding="utf-8")
        temp.replace(path)

    def companies(self) -> dict[str, Company]:
        from importlib.resources import files
        defaults = tomllib.loads(files("newscollector.resources").joinpath("companies.toml").read_text(encoding="utf-8"))
        result = {t: Company(ticker=t, **v) for t, v in defaults["companies"].items()}
        custom = self.root / "companies.json"
        if custom.exists():
            result.update({t: Company.model_validate(v) for t, v in json.loads(custom.read_text(encoding="utf-8")).items()})
        return result

    def save_company(self, company: Company) -> None:
        path = self.root / "companies.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        existing[company.ticker] = company.model_dump()
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        temp.replace(path)

    def source_profiles(self) -> dict[str, SourceProfile]:
        from importlib.resources import files
        data = tomllib.loads(files("newscollector.resources").joinpath("sources.toml").read_text(encoding="utf-8"))
        result = {key: SourceProfile.model_validate(value) for key, value in data["sources"].items()}
        path = self.root / "sources.json"
        if path.exists():
            result.update({key: SourceProfile.model_validate(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()})
        return result

    def save_source(self, key: str, profile: SourceProfile) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,49}", key):
            raise ValueError("Use a source ID such as independent_business.")
        path = self.root / "sources.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[key] = profile.model_dump()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)


def relevance(company: Company, title: str, body: str = "", scoped: bool = False) -> tuple[bool, str]:
    if scoped:
        return True, "Company-specific newsroom or filing source."
    # Navigation, related-story footers and long company lists must not grant relevance.
    lead = "\n".join(p for p in body.splitlines() if p.strip())[:2000]
    text = (title + "\n" + lead).casefold()
    for alias in [company.name, *company.aliases]:
        if len(alias) >= 3 and re.search(r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)", text):
            return True, f"Matched company name/alias: {alias}."
    if re.search(r"(?:\$|NASDAQ\s*:\s*|NYSE\s*:\s*)" + re.escape(company.ticker) + r"\b", title + lead, re.I):
        return True, f"Matched explicitly qualified ticker {company.ticker}."
    return False, "No unambiguous company name, configured alias or qualified ticker found."


def assess(company: Company, candidate: Candidate, title: str | None = None, body: str = "") -> dict:
    """Explainable eligibility; context does not assert a causal or price-direction claim."""
    good, reason = relevance(company, title or candidate.title, body, candidate.company_scoped)
    scope = "company" if good else candidate.metadata.get("scope", "company")
    if not good and scope == "sector":
        lead = ((title or candidate.title) + "\n" + body[:2000]).casefold()
        matched = [t for t in company.sector_terms if len(t) >= 3 and re.search(r"(?<!\w)" + re.escape(t.casefold()) + r"(?!\w)", lead)]
        for rule in company.context_rules:
            # A rule needs one geographical/entity anchor AND one subject term.
            has = lambda term: bool(re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", lead))
            if any(has(t) for t in rule.get("anchors", []) if len(t) >= 3) and any(has(t) for t in rule.get("topics", []) if len(t) >= 3):
                matched.append(rule.get("label", "Configured context rule"))
        good = bool(matched)
        reason = "Configured sector context term(s): " + ", ".join(matched) + "; not direct issuer news." if good else "No configured company or sector-context term in headline/lead."
    elif not good and scope == "market":
        good = True
        reason = "Explicitly configured macroeconomic context feed; not direct issuer news or a causal assertion."
    if not good:
        scope = "unmatched"
    headline = title or candidate.title
    recap = bool(re.search(r"\b(?:shares?|stock|ads)\b.{0,28}\b(?:rose|rises?|falls?|fell|drops?|dropped|gains?|gained|jumps?|jumped|slips?|slipped|surges?|surged|tumbles?|tumbled)\b", headline, re.I)
                 and not re.search(r"\b(?:after|because|amid|following|as|despite|on)\b", headline, re.I))
    flags = ["price_recap_candidate"] if recap else []
    if re.search(r"\b(?:price target|stock forecast|stocks? to buy|technical analysis)\b", headline, re.I):
        flags.append("investment_commentary")
    return {"accepted": good, "reason": reason, "scope": scope, "flags": flags,
            "method": "entity-context-rules/2", "role": candidate.metadata.get("role") or
            ("issuer" if candidate.company_scoped else "unclassified_reporting")}


@lru_cache(maxsize=16)
def _calendar(name: str):
    import exchange_calendars as xcals
    return xcals.get_calendar(name, start="2000-01-01", end="2035-12-31")


def session_for(value: str, calendar: str = "XNYS") -> str:
    import pandas as pd
    instant = pd.Timestamp(utc(value))
    cal = _calendar(calendar)
    session = cal.date_to_session(instant.date(), direction="next")
    if instant >= cal.session_close(session):
        session = cal.next_session(session)
    return str(session.date())
