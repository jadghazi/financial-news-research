import io
import json

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from newscollector.core import Candidate, Request, SourceProfile, now
from newscollector.extract import extract, linked_pdf
from newscollector.query import get_news
from newscollector.worker import enqueue, run_job, quarantine
from test_collection import fixture_fetcher
from test_temporal import article


def synthetic_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(b'BT /F1 12 Tf 40 700 Td (Test Robotics reported revenue of 120 million dollars.) Tj 0 -20 Td (The company did not expect demand to recover immediately.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(content)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_pdf_text_keeps_figures_and_negation_and_never_promotes_creation_date():
    result = extract(synthetic_pdf(), 'https://news.example.com/release.pdf', 'application/pdf')
    assert '120 million' in result['body'] and 'did not expect' in result['body']
    assert result['pdf_pages'] == 1 and result['published']['utc'] is None
    assert result['extractor'].startswith('pypdf-layout/')
    blank = PdfWriter()
    blank.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    blank.write(stream)
    with pytest.raises(ValueError, match='text layer'):
        extract(stream.getvalue(), 'https://example.org/scan.pdf')


def test_pdf_only_landing_is_not_a_full_article():
    raw = b'<html><article><p>This press release is available in PDF version only.</p><a href="https://data.alibabagroup.com/ecms-files/release.pdf">Download release</a></article></html>'
    url = 'https://www.alibabagroup.com/en-US/document-123'
    assert linked_pdf(raw, url) == 'https://data.alibabagroup.com/ecms-files/release.pdf'
    with pytest.raises(ValueError, match='landing page'):
        extract(raw, url)
    assert linked_pdf(raw, 'https://different.example.com/article') is None
    hostile = raw.replace(b'data.alibabagroup.com', b'localhost')
    with pytest.raises(ValueError, match='official PDF'):
        linked_pdf(hostile, url)


def test_pdf_release_pipeline_preserves_landing_and_pdf(store):
    company = store.settings.companies()['BABA']
    job = enqueue(store, Request(tickers=['BABA'], sources=['issuer'], since='2025-01-01', until='2025-12-31'))
    pdf = synthetic_pdf()
    def handler(request):
        if request.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nAllow: /\n')
        if request.url.path.endswith('ir-news-filings'):
            return httpx.Response(200, text='<a href="/en-US/document-123"><div class="component-title">Alibaba results</div><div class="date-date">October 2, 2025</div></a>')
        if request.url.path.endswith('.pdf'):
            return httpx.Response(200, content=pdf, headers={'content-type':'application/pdf'})
        return httpx.Response(200, text='<html><article><p>This press release is available in PDF version only.</p><a href="https://data.alibabagroup.com/ecms-files/release.pdf">Download</a></article></html>', headers={'content-type':'text/html'})
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    versions = store.rows("SELECT * FROM versions WHERE ticker='BABA'")
    assert len(versions) == 1
    meta = json.loads(versions[0]['metadata'])
    assert meta['content_scope'] == 'publisher_pdf'
    assert store.read_artifact(meta['source_evidence']['landing_artifact']).startswith(b'<html>')
    assert store.read_artifact(versions[0]['raw_hash']).startswith(b'%PDF')
    rows = get_news(store, 'BABA', now(), variant='full')['records']
    assert len(rows) == 1 and rows[0]['pdf_pages'] == 1
    assert rows[0]['content_url'].endswith('.pdf')


def test_quarantine_is_effective_from_recorded_check_time(store, job, company):
    _, vid = article(store, job, company)
    version = store.rows('SELECT raw_hash FROM versions WHERE id=?', (vid,))[0]
    quarantine(store, 'TST', version['raw_hash'], 'Known content-quality failure')
    assert get_news(store, 'TST', '2026-10-23T14:00:00Z', variant='full')['records']
    assert not get_news(store, 'TST', now(), variant='full')['records']
    assert get_news(store, 'TST', now(), variant='headline')['records']


def test_small_body_budget_includes_both_source_families(store, company):
    profile = SourceProfile(name='Independent fixture', url='https://news.example.com/independent', scope='company',
        role='independent_reporting', publisher_group='Independent', documentation='https://example.org/terms', rights_note='Synthetic')
    store.settings.save_source('independent', profile)
    company.source_ids = ['independent']
    store.settings.save_company(company)
    job = enqueue(store, Request(tickers=['TST'], sources=['feeds','publishers'], since='2026-10-01', until='2026-10-31', max_articles=2))
    requested = []
    def handler(request):
        requested.append(request.url.path)
        if request.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nAllow: /\n')
        if request.url.path in ('/feed','/independent'):
            name = request.url.path.strip('/')
            entries = ''.join(f'<item><title>Test Robotics update {name} {i}</title><link>https://news.example.com/{name}-{i}</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate></item>' for i in range(4))
            return httpx.Response(200, text=f'<rss><channel><title>{name}</title>{entries}</channel></rss>')
        return httpx.Response(200, text='<html><article><h1>Test Robotics update</h1><p>Test Robotics announced plans to build a factory. The company warned that demand remains uncertain and that financing conditions can affect its plans.</p></article></html>')
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    sources = {r['source'] for r in store.rows('SELECT source FROM versions')}
    assert sources == {'feeds','publishers'}
    assert len(store.rows("SELECT id FROM events WHERE stage='fetch_attempt'")) == 2
    assert len(store.rows('SELECT id FROM discoveries')) == 8


def test_resume_preserves_processing_budget_after_one_article(store, job):
    page_requests = []
    def handler(request):
        if request.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nAllow: /\n')
        if request.url.path == '/feed':
            entries = ''.join(f'<item><title>Test Robotics update {i}</title><link>https://news.example.com/item-{i}</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate></item>' for i in range(5))
            return httpx.Response(200, text=f'<rss><channel><title>News</title>{entries}</channel></rss>')
        page_requests.append(request.url.path)
        if len(page_requests) == 1:
            store.execute('UPDATE jobs SET cancel=1 WHERE id=?', (job,))
        return httpx.Response(200, text='<html><article><h1>Test Robotics update</h1><p>Test Robotics discussed its new manufacturing facility and warned that market demand remains uncertain. The company plans to update investors about progress over the coming months.</p></article></html>')
    factory = fixture_fetcher(handler)
    run_job(store, job, fetcher_factory=factory)
    assert store.job(job)['status'] == 'cancelled' and len(page_requests) == 1
    store.execute("UPDATE jobs SET cancel=0,status='queued' WHERE id=?", (job,))
    run_job(store, job, fetcher_factory=factory)
    assert len(page_requests) == 3  # the fixture's original whole-job budget
    assert len(store.rows("SELECT id FROM discoveries WHERE reason LIKE '%fetch limit%'")) == 2
