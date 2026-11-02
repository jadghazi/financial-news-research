from pathlib import Path

from streamlit.testing.v1 import AppTest

from newscollector.demo import load_demo
from newscollector.query import get_news
from newscollector.core import now


def test_offline_demo_is_idempotent_and_exports(store):
    first = load_demo(store)
    assert load_demo(store) == first
    data = get_news(store, "DEMO", "2026-10-24T12:00:00Z", variant="full")
    assert len(data["records"]) == 2
    assert len(data["duplicates_suppressed"]) == 1
    assert store.rows("SELECT COUNT(*) n FROM discoveries")[0]["n"] == 5


def test_every_dashboard_page_and_review(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_COLLECTOR_HOME", str(tmp_path / "ui"))
    app_path = Path(__file__).parents[1] / "src/newscollector/dashboard.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    app.sidebar.selectbox[0].select("Offline demo").run()
    assert not app.exception
    for page in ["Headlines", "Predict", "History", "Settings"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {list(app.exception)}"
    for page in ["Model", "Overview", "Sources", "Companies", "Corpus quality", "Audit"]:
        app.sidebar.selectbox[-1].set_value(page).run()
        assert not app.exception, f"{page}: {list(app.exception)}"
    app.sidebar.selectbox[-1].set_value("Corpus quality").run()
    next(b for b in app.button if b.label == "Create frozen relevance audit").click().run()
    assert not app.exception
    app.sidebar.selectbox[-1].set_value("Audit").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Save audit review").click().run()
    assert not app.exception
    assert any("saved" in s.value.lower() for s in app.success)
