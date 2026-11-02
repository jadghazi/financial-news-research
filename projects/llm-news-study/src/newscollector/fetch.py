"""One bounded, robots-aware HTTP path. No extractor makes hidden requests."""
from __future__ import annotations

import ipaddress
import json
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .core import now
from .store import Store


class FetchError(RuntimeError):
    pass


@dataclass
class Fetched:
    url: str
    body: bytes
    artifact: str
    at: str
    content_type: str
    status: int


def public_url(url: str):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise FetchError("Only public HTTP/HTTPS URLs without embedded credentials are allowed.")
    if parsed.port not in (None, 80, 443):
        raise FetchError("Custom URL ports are not enabled for external news sources.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"DNS lookup failed for {parsed.hostname}.") from exc
    if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
        raise FetchError("Local, private and reserved network addresses are blocked.")


class Fetcher:
    def __init__(self, store: Store, job: str, *, client: httpx.Client | None = None,
                 validate=public_url, sleep=time.sleep):
        self.store, self.job, self.validate, self.sleep = store, job, validate, sleep
        self.owned = client is None
        self.client = client or httpx.Client(timeout=httpx.Timeout(20, connect=10),
                                            headers={"User-Agent": store.settings.user_agent},
                                            follow_redirects=False, limits=httpx.Limits(max_connections=2))
        self.robots: dict[str, RobotFileParser] = {}
        self.last: dict[str, float] = {}
        self.delays: dict[str, float] = {}

    def close(self):
        if self.owned:
            self.client.close()

    def _paced(self, url: str, delay: float):
        host = urlsplit(url).hostname
        remaining = max(delay, self.delays.get(host, 0)) - (time.monotonic() - self.last.get(host, 0))
        if remaining > 0:
            self.sleep(remaining)
        self.last[host] = time.monotonic()

    def _request(self, url: str, delay: float = 1.0, *, allowed_error: bool = False) -> Fetched:
        start_url = url
        chain = []
        for _ in range(6):
            self.validate(url)
            self.store.checkpoint(self.job)
            self._paced(url, delay)
            for attempt in range(3):
                try:
                    started = now()
                    with self.client.stream("GET", url) as response:
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > self.store.settings.max_bytes:
                                raise FetchError("Response exceeds the configured 8 MB safety limit.")
                            chunks.append(chunk)
                        body = b"".join(chunks)
                        status, headers = response.status_code, dict(response.headers)
                    at = now()
                    artifact = self.store.archive(body)
                    detail = {"started_at": started, "finished_at": at, "headers": headers,
                              "attempt": attempt + 1, "redirect_chain": chain, "requested_url": start_url}
                    self.store.execute("INSERT INTO fetch_log(job_id,url,at,status,artifact,detail) VALUES(?,?,?,?,?,?)",
                                       (self.job, url, at, status, artifact, json.dumps(detail)))
                    if status == 429 or status in (500, 502, 503, 504):
                        wait = min(60.0, 2.0 ** (attempt + 1))
                        retry = headers.get("retry-after", "")
                        try:
                            wait = max(wait, float(retry))
                        except ValueError:
                            try:
                                wait = max(wait, (parsedate_to_datetime(retry).astimezone(UTC) - datetime.now(UTC)).total_seconds())
                            except (ValueError, TypeError):
                                pass
                        self.store.event(self.job, "retry", f"HTTP {status}; attempt {attempt+1}/3.", "warning", url=url)
                        if attempt == 2 or wait > 60:
                            raise FetchError(f"HTTP {status}; retry budget exhausted or Retry-After exceeds this run's wait budget.")
                        self.sleep(wait)
                        self.store.checkpoint(self.job)
                        continue
                    break
                except httpx.HTTPError as exc:
                    if attempt == 2:
                        raise FetchError(f"Network failure after 3 attempts: {type(exc).__name__}.") from exc
                    self.store.event(self.job, "retry", f"{type(exc).__name__}; retrying.", "warning", url=url)
                    self.sleep(2 ** (attempt + 1))
            if status in (301, 302, 303, 307, 308):
                target = urljoin(url, headers.get("location", ""))
                if target == url or target in chain:
                    raise FetchError("Redirect loop detected.")
                chain.append(url)
                # Every redirected destination receives its own robots check.
                if not allowed_error and not urlsplit(target).path.endswith("/robots.txt"):
                    self._check_robots(target)
                url = target
                continue
            if status >= 400 and not allowed_error:
                raise FetchError(f"HTTP {status}: access unavailable; no bypass attempted.")
            return Fetched(url, body, artifact, at, headers.get("content-type", ""), status)
        raise FetchError("Too many redirects.")

    def _check_robots(self, url: str):
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            r = self._request(origin + "/robots.txt", allowed_error=True)
            parser = RobotFileParser(origin + "/robots.txt")
            if r.status in (404, 410):
                parser.parse([])
            elif r.status != 200:
                raise FetchError(f"robots.txt returned HTTP {r.status}; access not attempted.")
            else:
                parser.parse(r.body.decode("utf-8", "replace").splitlines())
            self.robots[origin] = parser
            self.delays[parsed.hostname] = float(parser.crawl_delay("NewsStudyCollector") or parser.crawl_delay("*") or 1)
        if not self.robots[origin].can_fetch("NewsStudyCollector", url):
            raise FetchError("robots.txt disallows this URL for the collector.")

    def get(self, url: str, *, delay: float = 1.0) -> Fetched:
        self._check_robots(url)
        return self._request(url, delay)
