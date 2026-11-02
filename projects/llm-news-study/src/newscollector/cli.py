"""Professor-friendly commands; all data stays in the selected local workspace."""
from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .core import Company, Request, Settings, now
from .query import export_news, get_news
from .sources import REGISTRY
from .store import Store
from .worker import enqueue, work

app = typer.Typer(no_args_is_help=True, help="Collect ticker news, inspect evidence, and export structured text. No LLM required.")


def store_at(home: Path) -> Store:
    return Store(Settings(home))


@app.command()
def doctor(home: Path = Path("data")):
    """Check the local workspace, SQLite mode and available source configuration."""
    store = store_at(home)
    journal = store.rows("PRAGMA journal_mode")[0]["journal_mode"]
    typer.echo(json.dumps({"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
        "journal_mode": journal, "workspace": str(store.settings.root), "companies": list(store.settings.companies()),
        "sec_contact_configured": bool(store.settings.contact), "database_integrity": store.rows("PRAGMA integrity_check"),
        "scope": "Collector CLI works independently; the dashboard offers optional headline experiments with separate scoring."}, indent=2))


@app.command()
def sources():
    """List source capabilities and their coverage/access limits."""
    typer.echo(json.dumps(REGISTRY, indent=2))


@app.command("add-company")
def add_company(ticker: str, name: str, feed: list[str] = typer.Option([], "--feed"),
                alias: list[str] = typer.Option([], "--alias"), cik: str | None = None,
                company_feed: bool = False,
                home: Path = Path("data")):
    """Configure another ticker without changing source code."""
    Settings(home).save_company(Company(ticker=ticker, name=name, feeds=feed, aliases=alias, cik=cik,
                                        feed_scope="company" if company_feed else "general"))
    typer.echo(f"Saved {ticker.upper()}. Check publisher access conditions before collecting.")


def _collect(tickers, source, since, until, limit, home, watch=False, interval=15, headlines_only=False):
    try:
        store = store_at(home)
        request = Request(tickers=tickers.split(","), sources=source.split(","),
                          since=since, until=until, max_articles=limit,
                          mode="watch" if watch else "once", interval_minutes=interval, collect_bodies=not headlines_only)
        job = enqueue(store, request)
        typer.echo(f"Queued {job}. Source events and results are stored in {store.settings.root}.")
        work(store, once=not watch)
        result = store.job(job)
        typer.echo(json.dumps({"job_id": job, "status": result["status"], "summary": json.loads(result["summary"]), "error": result["error"]}, indent=2))
        if result["status"] == "failed":
            raise typer.Exit(1)
    except (ValueError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)


@app.command()
def collect(tickers: str = "AAPL", source: str = "feeds",
            since: str = str(date.today() - timedelta(days=30)), until: str = str(date.today()),
            limit: int = 20, home: Path = Path("data"), headlines_only: bool = False):
    """Collect once. --tickers AAPL,MSFT --source feeds,sec. Dates are publication dates."""
    _collect(tickers, source, since, until, limit, home, headlines_only=headlines_only)


@app.command()
def backfill(tickers: Annotated[str, typer.Option("--tickers")], since: Annotated[str, typer.Option("--since")],
             until: Annotated[str, typer.Option("--until")], source: str = "feeds", limit: int = 100, home: Path = Path("data"), headlines_only: bool = False):
    """Request a historical interval; coverage limitations are explicitly reported."""
    _collect(tickers, source, since, until, limit, home, headlines_only=headlines_only)


@app.command()
def watch(tickers: str = "AAPL", source: str = "feeds", interval: int = 15, limit: int = 20, home: Path = Path("data")):
    """Poll with durable jobs and seven-day reconciliation. Ctrl+C stops this worker."""
    _collect(tickers, source, str(date.today() - timedelta(days=7)), str(date.today()), limit, home, True, interval)


@app.command()
def resume(job_id: str, home: Path = Path("data")):
    """Resume pending items from an interrupted or cancelled job."""
    store = store_at(home)
    store.job(job_id)
    store.execute("UPDATE jobs SET status='queued',cancel=0,error=NULL WHERE id=?", (job_id,))
    work(store, once=True)
    typer.echo(store.job(job_id)["status"])


@app.command()
def demo(home: Path = Path("data/demo")):
    """Load synthetic offline examples in a separate workspace."""
    from .demo import load_demo
    store = store_at(home)
    existing = store.rows("SELECT id FROM jobs WHERE json_extract(request,'$.demo_loaded')=1")
    job = existing[0]["id"] if existing else load_demo(store)
    typer.echo(f"Synthetic demo ready: {job}. Export with --home {home} --ticker DEMO.")


@app.command()
def export(ticker: Annotated[str, typer.Option("--ticker")], output: Annotated[Path, typer.Option("--output")], variant: str = "headline", as_of: str | None = None,
           policy: str = "observed", format: str = "jsonl", since: str | None = None,
           until: str | None = None, max_chars: int | None = None, scopes: str = "company",
           include_price_recaps: bool = False, cohort: str = "available", home: Path = Path("data"), include_social: bool = False):
    """Export headlines, lead or full text with a companion manifest. Never overwrite."""
    try:
        result = get_news(store_at(home), ticker, as_of or now(), variant=variant, policy=policy,
                          since=since, until=until, max_chars=max_chars, scopes=scopes.split(","),
                          include_price_recaps=include_price_recaps, cohort=cohort, include_social=include_social)
        path, meta = export_news(result, output, format)
        typer.echo(f"Exported {len(result['records'])} records.\nData: {path}\nManifest: {meta}\n{result['warning']}")
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)


@app.command()
def quality(ticker: Annotated[str, typer.Option("--ticker")], since: str | None = None,
            until: str | None = None, home: Path = Path("data")):
    """Report observed corpus origins, text availability, timing and quality limitations."""
    from .quality import corpus_report
    typer.echo(json.dumps(corpus_report(store_at(home), ticker, since=since, until=until), indent=2))


@app.command("audit-create")
def audit_create(ticker: Annotated[str, typer.Option("--ticker")], output: Annotated[Path, typer.Option("--output")],
                 per_group: int = 25, seed: int = 20261024, since: str | None = None,
                 until: str | None = None, home: Path = Path("data")):
    """Freeze a random relevance-review sample with its population and selection method."""
    from .quality import create_audit
    if output.exists():
        raise typer.BadParameter("Choose a new audit filename; existing evidence is preserved.")
    spec = create_audit(store_at(home), ticker, per_group=per_group, seed=seed, since=since, until=until)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, ensure_ascii=False)
    typer.echo(f"Frozen audit {spec['audit_id']}: {len(spec['records'])} items. Review in the dashboard's Audit page.")


@app.command()
def reextract(record_id: str, home: Path = Path("data")):
    """Re-extract saved bytes without fetching the publisher again."""
    from .fetch import Fetched
    from .worker import ingest
    store = store_at(home)
    rows = store.rows("SELECT * FROM versions WHERE id=?", (record_id,))
    if not rows:
        raise typer.BadParameter("Unknown article version; copy its record_id from an export or inspector.")
    row = rows[0]
    meta = json.loads(row["metadata"])
    company = Company.model_validate(meta["company_config"])
    result = ingest(store, meta["discovery_id"], company,
                    Fetched(row["url"], store.read_artifact(row["raw_hash"]), row["raw_hash"], row["fetched_at"], "text/html", 200))
    typer.echo(f"Extraction version: {result}. Previous versions are preserved; unchanged extraction is reused.")


@app.command()
def status(home: Path = Path("data")):
    """Print recent jobs and source failures."""
    store = store_at(home)
    typer.echo(json.dumps({"jobs": store.rows("SELECT id,status,created_at,summary,error FROM jobs ORDER BY created_at DESC LIMIT 15"),
                           "source_failures": store.rows("SELECT ticker,source,detail FROM source_runs WHERE status='failed' ORDER BY id DESC LIMIT 10")}, indent=2))


@app.command()
def stop_worker(home: Path = Path("data")):
    """Stop the background collector after its current run; preserve future schedules."""
    settings = Settings(home)
    (settings.root / "stop-worker").write_text("stop after current run\n", encoding="utf-8")
    typer.echo("Worker stop requested. Existing data and pending schedules are preserved.")


@app.command()
def verify(home: Path = Path("data")):
    """Check database integrity and every archived checksum without refetching."""
    store = store_at(home)
    errors, checked = [], 0
    for path in store.artifacts.glob("*/*.gz"):
        try:
            store.read_artifact(path.stem)
            checked += 1
        except Exception as exc:
            errors.append({"artifact": path.stem, "error": str(exc)})
    integrity = store.rows("PRAGMA integrity_check")
    typer.echo(json.dumps({"checked_artifacts": checked, "errors": errors, "database": integrity}, indent=2))
    if errors or integrity != [{"integrity_check": "ok"}]:
        raise typer.Exit(1)


@app.command()
def backup(destination: Path, home: Path = Path("data")):
    """Save a database and artifact-store backup outside the active data folder."""
    store_at(home).backup(destination)
    typer.echo(f"Backup saved to {destination.resolve()}")


@app.command()
def dashboard(home: Path = Path("data"), port: int = 8501, open_browser: bool = True):
    """Open the local dashboard. Launch.cmd provides the same entry point on Windows."""
    from importlib.resources import files
    env = dict(os.environ, NEWS_COLLECTOR_HOME=str(home.resolve()))
    script = str(Path(__file__).with_name("dashboard.py"))
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", script,
                                     "--server.address", "127.0.0.1", "--server.port", str(port),
                                     "--server.headless", "true", "--browser.gatherUsageStats", "false",
                                     "--theme.base", "light", "--theme.primaryColor", "#2563eb"], env=env)
    if open_browser:
        import time
        import urllib.request
        import webbrowser
        for _ in range(30):
            if process.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).close()
                webbrowser.open(f"http://127.0.0.1:{port}")
                break
            except OSError:
                time.sleep(0.5)
    try:
        raise typer.Exit(process.wait())
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    app()
