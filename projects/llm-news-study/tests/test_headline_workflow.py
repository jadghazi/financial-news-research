"""Headline discovery must not depend on article access or lose source identities."""
import json

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from newscollector.broad_sources import aggregator_candidates, regulatory_candidates, social_candidates
from newscollector.core import Candidate, Request, assess, now
from newscollector.query import get_news
from newscollector.worker import discover, enqueue, run_job
from test_collection import fixture_fetcher


def test_headlines_only_never_requests_article_and_ignores_body_budget(store, company):
    store.settings.save_company(company)
    job = enqueue(store, Request(tickers=[company.ticker], sources=['feeds'], since='2026-10-01', until='2026-10-31',
                                 collect_bodies=False, max_articles=1))
    requested=[]
    def handler(r):
        requested.append(r.url.path)
        if r.url.path == '/robots.txt':
            return httpx.Response(200,text='User-agent: *\nAllow: /')
        if r.url.path == '/feed':
            return httpx.Response(200,text='<rss><channel><title>Fixture feed</title>' + ''.join(
                f'<item><title>Test Robotics announcement {i}</title><link>https://news.example.com/blocked-{i}</link>'
                '<pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate></item>' for i in range(5)) + '</channel></rss>')
        pytest.fail('Headline collection must not visit article pages')
    run_job(store,job,fetcher_factory=fixture_fetcher(handler))
    assert store.job(job)['status']=='completed'
    assert len(get_news(store,'TST',now())['records'])==5
    assert len(store.rows("SELECT id FROM discoveries WHERE state='headline_only'"))==5
    assert not store.rows('SELECT id FROM versions')
    assert not any(p.startswith('/blocked') for p in requested)


def test_aggregator_parses_dated_news_cards_not_price_data(store,job,company):
    fixture = '<html><h3><a href="https://ads.example.com/">Advertisement</a></h3><div><h3><a href="https://reporter.example.com/story">Test Robotics earnings rise</a></h3><div title="2026-10-23T11:00:00Z">2 days ago - Independent Daily</div><p>Price target 123</p></div></html>'
    def handler(r):
        return httpx.Response(200,text='User-agent: *\nAllow: /' if r.url.path=='/robots.txt' else fixture)
    f=fixture_fetcher(handler)(store,job)
    items,note,raw=aggregator_candidates(f,company,Request(tickers=['TST'],since='2026-10-01',until='2026-10-31'))
    assert len(items)==1 and items[0].publisher=='Independent Daily'
    assert items[0].url=='https://reporter.example.com/story'
    assert items[0].published['semantic']=='aggregator_reported_publication'
    assert items[0].provider_time['semantic']=='provider_reported_publication_not_receipt'
    assert '123' not in items[0].model_dump_json() and 'INCOMPLETE' in note


def test_regulatory_pagination_is_bounded_and_keeps_date_only(store,job,company):
    company.regulatory_queries=['China']
    pages=[]
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(200,text='User-agent: *\nAllow: /')
        pages.append(r.url.params['page'])
        return httpx.Response(200,json={'results':[{'html_url':'https://www.federalregister.gov/story/'+pages[-1],
                    'title':'China technology export controls','publication_date':'2025-09-01','document_number':pages[-1],
                    'agencies':[{'name':'Commerce Department'}]}], 'total_pages':3,'next_page_url':'https://evil.invalid/not-followed'})
    items,note,raw=regulatory_candidates(fixture_fetcher(handler)(store,job),company,
        Request(tickers=['TST'],since='2025-01-01',until='2025-12-31',discovery_pages=2))
    assert pages==['1','2'] and len(items)==2
    assert all(x.published['utc'] is None and x.kind=='regulatory_notice' for x in items)
    assert 'INCOMPLETE' in note


def test_broader_company_rules_need_an_anchor_and_subject(store):
    baba=store.settings.companies()['BABA']
    base=dict(url='https://example.org/',source='publishers',publisher='Fixture',metadata={'scope':'sector'})
    assert assess(baba,Candidate(title='China tightens cloud export controls',**base))['scope']=='sector'
    assert not assess(baba,Candidate(title='China wins a sporting contest',**base))['accepted']
    assert not assess(baba,Candidate(title='Canadian consumer enjoys cloud pictures',**base))['accepted']
    assert assess(baba,Candidate(title='Qwen introduces new model',**base))['scope']=='company'


def test_social_pinned_identity_no_reposts_and_opt_in_export(store,job,company):
    company.social_accounts=[{'handle':'trusted.example','did':'did:plc:fixture','name':'Trusted Fixture','identity_evidence':'Synthetic'}]
    def post(did='did:plc:fixture',**extra):
        return {'post':{'author':{'did':did,'handle':'trusted.example'},'uri':f'at://{did}/app.bsky.feed.post/one',
                'record':{'createdAt':'2026-10-23T12:00:00Z','text':'Test Robotics updates its factory plans'},
                'indexedAt':'2026-10-23T12:00:01Z','likeCount':999999,'repostCount':888888},**extra}
    def handler(r):
        return httpx.Response(200,text='User-agent: *\nAllow: /') if r.url.path=='/robots.txt' else httpx.Response(200,json={'feed':[post(),post(reason={'$type':'repost'}),post(did='did:plc:impostor')]})
    items,_,_=social_candidates(fixture_fetcher(handler)(store,job),company,
                              Request(tickers=['TST'],since='2026-10-01',until='2026-10-31'))
    assert len(items)==1
    discover(store,job,123,'TST',company,items[0],seen='2026-10-24T12:00:00Z')
    assert not get_news(store,'TST',now())['records']
    rows=get_news(store,'TST',now(),include_social=True)['records']
    assert len(rows)==1 and rows[0]['kind']=='social_post'
    assert rows[0]['social_provenance']['did']=='did:plc:fixture'
    assert '999999' not in json.dumps(rows) and 'likeCount' not in json.dumps(rows)
    assert not get_news(store,'TST','2026-10-23T13:00:00Z',include_social=True)['records']


def test_new_ui_has_headline_default_clear_controls_and_navigation(tmp_path,monkeypatch):
    monkeypatch.setenv('NEWS_COLLECTOR_HOME',str(tmp_path/'ui'))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/newscollector/dashboard.py'),default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].options==['Headlines','Predict','History','Settings']
    assert any(b.label=='Find headlines' for b in app.button)
    assert not any('full article' in c.label.lower() for c in app.checkbox)
    assert {'archives','news_sites','cloud'} <= set(next(x for x in app.multiselect if x.label=='Sources to search').value)
    app.sidebar.radio[0].set_value('Predict').run()
    assert not app.exception
    assert any('No saved headlines yet' in x.value for x in app.info)


def test_recorded_headline_check_preserves_earlier_observed_exports(store,job,company):
    c=Candidate(url='https://example.org/checked',title='Test Robotics updates',publisher='Fixture',source='fixture')
    key=discover(store,job,321,'TST',company,c,seen='2026-10-22T12:00:00Z')
    store.execute('INSERT INTO headline_quarantines(ticker,discovery_id,at,reason) VALUES(?,?,?,?)',
                  ('TST',key,'2026-10-24T12:00:00.000000+00:00','Unrelated regulatory search result in pilot'))
    assert len(get_news(store,'TST','2026-10-23T12:00:00Z')['records'])==1
    checked=get_news(store,'TST','2026-10-25T12:00:00Z')
    assert not checked['records'] and 'Recorded headline quality check' in checked['exclusions'][0]['reason']


@pytest.mark.parametrize('url',['javascript:alert(1)','https://user:secret@example.org/','not-a-link'])
def test_headline_only_export_rejects_unsupported_source_links(store,job,company,url):
    c=Candidate(url=url,title='Test Robotics update',publisher='Fixture',source='fixture')
    discover(store,job,333,'TST',company,c,seen='2026-10-23T12:00:00Z')
    assert not get_news(store,'TST',now())['records']


def test_headline_dedup_keeps_recurring_events_and_distinct_regulatory_notices(store,job,company):
    from newscollector.core import timestamp
    for i,(day,kind) in enumerate([('2026-10-21','reporting'),('2026-10-22','reporting'),
                                  ('2026-10-22','reporting'),('2026-10-23','regulatory_notice'),
                                  ('2026-10-23','regulatory_notice')]):
        c=Candidate(url=f'https://example.org/repeated-{i}',title='Test Robotics receives a regulatory update',
                    publisher='Fixture',source='fixture',kind=kind,published=timestamp(day))
        discover(store,job,444,'TST',company,c,seen='2026-10-24T12:00:00Z')
    result=get_news(store,'TST',now())
    assert len(result['records'])==4
    assert len(result['duplicates_suppressed'])==1
