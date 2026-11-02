import pytest

from newscollector.core import Company, Request, Settings
from newscollector.store import Store
from newscollector.worker import enqueue


@pytest.fixture
def store(tmp_path):
    return Store(Settings(tmp_path / "data"))


@pytest.fixture
def company():
    return Company(ticker="TST", name="Test Robotics", aliases=["Test Robotics"], feeds=["https://news.example.com/feed"])


@pytest.fixture
def job(store, company):
    store.settings.save_company(company)
    return enqueue(store, Request(tickers=["TST"], since="2026-10-01", until="2026-10-31", max_articles=3))
