import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from newscollector.archive_sources import archive_candidates, cloud_candidates, cnbc_metadata, news_site_candidates
from newscollector.core import Candidate, Request, now, publication_day, timestamp
from newscollector.query import get_news
from newscollector.worker import discover, enqueue, run_job
from test_collection import fixture_fetcher


def candidate(company, suffix, day, title=None):
    return Candidate(url=f'https://news.example.com/{suffix}', title=title or f'Test Robotics {suffix}', publisher='Fixture',
                     source='feeds',published=timestamp(day),company_scoped=True)


def test_exact_run_filter_precedes_url_dedup_and_dates_are_new_york(store, company):
    store.settings.save_company(company)
    req=Request(tickers=['TST'],since='2025-09-27',until='2025-10-16',collect_bodies=False)
    first, second=enqueue(store,req),enqueue(store,req)
    for job in [first,second]:
        for suffix,day in [('inside','2025-10-16T23:00:00Z'),('night','2025-10-17T01:00:00Z'),
                           ('target','2025-10-17T05:00:00Z'),('old','2024-01-01'),('new','2026-01-01')]:
            discover(store,job,1,'TST',company,candidate(company,suffix,day))
    result=get_news(store,'TST',now(),since='2025-09-27',until='2025-10-16',job_id=second)
    assert len(result['records'])==2
    assert {r['collection_id'] for r in result['records']}=={second}
    assert {r['publication_date'] for r in result['records']}=={'2025-10-16'}
    assert result['query']['job_id']==second
    with pytest.raises(ValueError,match='saved collection'):get_news(store,'BABA',now(),job_id=second)
    assert publication_day(timestamp('2025-10-17T01:00:00Z'))=='2025-10-16'
    assert publication_day(timestamp('2025-10-17'))=='2025-10-17'


def test_out_of_range_discoveries_remain_auditable_but_are_not_results(store,company):
    store.settings.save_company(company)
    job=enqueue(store,Request(tickers=['TST'],since='2025-10-01',until='2025-10-16',collect_bodies=False))
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(200,text='User-agent: *\nAllow: /')
        return httpx.Response(200,text='<rss><channel><title>Fixture</title>'+''.join(
            f'<item><title>Test Robotics {d}</title><link>https://news.example.com/{i}</link><pubDate>{d}</pubDate></item>'
            for i,d in enumerate(['2024-01-01','2025-10-10','2026-01-01']))+'</channel></rss>')
    run_job(store,job,fetcher_factory=fixture_fetcher(handler))
    assert len(store.rows('SELECT * FROM discoveries WHERE job_id=?',(job,)))==3
    assert len(get_news(store,'TST',now(),job_id=job,since='2025-10-01',until='2025-10-16')['records'])==1
    assert len(store.rows("SELECT * FROM discoveries WHERE job_id=? AND reason LIKE 'Outside%'",(job,)))==2


def test_public_archives_follow_only_bounded_pagination_and_never_article_pages(store,job,company):
    calls=[]
    def handler(r):
        calls.append(r.url.path)
        if r.url.path=='/robots.txt':return httpx.Response(200,text='User-agent: *\nAllow: /')
        if r.url.path not in ['/2025/10/','/2025/10/page/2/']:pytest.fail('Archive must not fetch article pages')
        day='2025-10-16T14:00:00+08:00' if r.url.path=='/2025/10/' else '2025-09-01T14:00:00+08:00'
        return httpx.Response(200,text=f'<article class="type-post"><h2 class="entry-title"><a href="https://technode.com/story/{len(calls)}">Test Robotics update</a></h2><time class="published" datetime="{day}"></time><time class="updated" datetime="2026-10-01T12:00:00Z"></time></article><a class="next page-numbers" href="https://technode.com/2025/10/page/2/">Next</a>')
    items,note,_=archive_candidates(fixture_fetcher(handler)(store,job),company,Request(tickers=['TST'],since='2025-10-01',until='2025-10-16',discovery_pages=2))
    assert len(items)==2 and len(calls)==3
    assert items[0].published['raw']=='2025-10-16T14:00:00+08:00'
    assert all(x.metadata['body_mode']=='metadata_only' for x in items)


def test_cloud_dates_and_metadata_never_invented(store,job):
    company=store.settings.companies()['BABA']
    data=[{'contentArrItem':[{'heading':'Alibaba Cloud update','date':'September 22','year':'2025','readMoreLink':'https://www.alibabacloud.com/press-room/update'}]}]
    def handler(r):return httpx.Response(200,text='User-agent: *\nAllow: /' if r.url.path=='/robots.txt' else '<script>window.pressPageData["tab0"]='+json.dumps(data)+';</script>')
    items,_,_=cloud_candidates(fixture_fetcher(handler)(store,job),company,Request(tickers=['BABA'],since='2025-09-01',until='2025-10-16'))
    assert len(items)==1 and publication_day(items[0].published)=='2025-09-22'
    assert items[0].metadata['role']=='issuer'


def test_cnbc_uses_public_headline_metadata_even_when_body_is_subscription_flagged():
    text='<meta property="og:title" content="Alibaba &amp; partners announce an expansion"><meta property="article:published_time" content="2025-10-16T12:00:00Z">'
    def result(t):return SimpleNamespace(body=t.encode(),url='https://www.cnbc.com/2025/10/16/alibaba.html',artifact='fixture')
    c=cnbc_metadata(result(text))
    assert c.title=='Alibaba & partners announce an expansion'
    assert c.metadata['body_mode']=='metadata_only'
    with pytest.raises(ValueError,match='Missing explicit'):cnbc_metadata(result('<title>Not publication evidence</title>'))
    public_headline=cnbc_metadata(result(text+'<script type="application/ld+json">{"isAccessibleForFree":false}</script>'))
    assert public_headline.title==c.title and public_headline.metadata['body_access']=='subscription_flagged'
    assert public_headline.metadata['body_mode']=='metadata_only'


def test_results_ui_is_latest_search_only_clear_preserves_data_and_prediction_handoff(store,company,monkeypatch):
    store.settings.save_company(company)
    for title,day in [('Old search headline','2024-10-16'),('Latest matching headline','2025-10-16')]:
        job=enqueue(store,Request(tickers=['TST'],since=day,until=day,collect_bodies=False))
        discover(store,job,1,'TST',company,candidate(company,title.replace(' ','-'),day,title='Test Robotics '+title))
        discover(store,job,1,'TST',company,candidate(company,'outside-'+day,'2026-01-01',title='Test Robotics outside window'))
        store.execute("UPDATE jobs SET status='completed' WHERE id=?",(job,))
    monkeypatch.setenv('NEWS_COLLECTOR_HOME',str(store.settings.root))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/newscollector/dashboard.py'),default_timeout=40).run()
    app.sidebar.selectbox[1].set_value('TST').run()
    assert not app.exception
    table=' '.join(m.value for m in app.markdown if '<table class="news-table">' in m.value)
    assert 'Latest matching headline' in table and 'Old search headline' not in table and 'outside window' not in table
    next(x for x in app.button if x.label=='Use these headlines →').click().run()
    assert not app.exception
    assert next(x for x in app.date_input if x.label=='Prediction date').value==date(2025,10,17)
    text=next(x for x in app.text_area if x.label=='Exact dated-headline input').value
    assert 'Latest matching headline' in text and 'outside window' not in text and 'Old search headline' not in text
    app.sidebar.radio[0].set_value('Headlines').run()
    next(x for x in app.button if x.label=='Clear view').click().run()
    assert not app.exception and not any('<table class="news-table">' in m.value for m in app.markdown)
    assert len(store.rows('SELECT * FROM discoveries'))==4


def test_html_headline_output_escapes_untrusted_markup(monkeypatch):
    from newscollector import workspace
    output=[]
    monkeypatch.setattr(workspace.st,'markdown',lambda value,**kwargs:output.append(value))
    workspace.headline_table([{'date':'2025-10-16','headline':'<script>alert("x")</script>','url':'javascript:alert(1)'}])
    markup=' '.join(output)
    assert '<script>' not in markup and '&lt;script&gt;' in markup and 'javascript:' not in markup


def test_dashboard_find_button_runs_headline_only_and_shows_only_requested_dates(store,company,monkeypatch):
    from newscollector import workspace
    store.settings.save_company(company)
    monkeypatch.setenv('NEWS_COLLECTOR_HOME',str(store.settings.root))
    requests=[]
    def handler(r):
        requests.append(r.url.path)
        if r.url.path=='/robots.txt':return httpx.Response(200,text='User-agent: *\nAllow: /')
        if r.url.path!='/feed':pytest.fail('Headline-only search must not fetch article bodies')
        return httpx.Response(200,text='<rss><channel><title>Fixture</title>'+''.join(
            f'<item><title>Test Robotics {day}</title><link>https://news.example.com/story-{i}</link><pubDate>{day}</pubDate></item>'
            for i,day in enumerate(['2024-01-01','2025-10-16','2026-01-01']))+'</channel></rss>')
    def run(settings,once):
        key=store.rows("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at DESC")[0]['id']
        run_job(store,key,fetcher_factory=fixture_fetcher(handler))
        return 123
    monkeypatch.setattr(workspace,'start_worker',run)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/newscollector/dashboard.py'),default_timeout=40).run()
    app.sidebar.selectbox[1].set_value('TST').run()
    next(x for x in app.date_input if x.label=='News published from').set_value(date(2025,9,27))
    next(x for x in app.date_input if x.label=='Through').set_value(date(2025,10,16))
    next(x for x in app.multiselect if x.label=='Sources to search').set_value(['feeds'])
    next(x for x in app.button if x.label=='Find headlines').click().run()
    assert not app.exception
    rows=store.rows('SELECT request,status FROM jobs')
    assert len(rows)==1 and rows[0]['status']=='completed' and json.loads(rows[0]['request'])['collect_bodies'] is False
    markup=' '.join(m.value for m in app.markdown if '<table class="news-table">' in m.value)
    assert '2025-10-16' in markup and '2024-01-01' not in markup and '2026-01-01' not in markup
    assert requests==['/robots.txt','/feed']
