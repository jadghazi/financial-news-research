from datetime import date
from pathlib import Path
import io
import json
import zipfile

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from newscollector import RUNTIME_CODE_SHA256
from newscollector.core import Settings, digest
from newscollector.experiments import (ExperimentStore, block_interval, bundle, create_study,
    labels_from_csv, normalize_headline, packet, parse_headlines, report, save_labels, trading_dates)
from newscollector.experiment_worker import pause, work
from newscollector.gemma import GemmaClient, MODEL, ProviderError, SYSTEM, parse_prediction, request_body


@pytest.fixture
def exp(tmp_path):
    (tmp_path/'google-credentials.json').write_text(json.dumps({'api_key':'fixture-secret-only'}))
    return ExperimentStore(tmp_path)


def headlines():
    return parse_headlines('2025-10-02 -- Old product update\n2025-10-12 -- Weekend partnership\n2025-10-16 -- Company announces results\n2025-10-17 -- Target day headline must not appear')


def test_dated_input_formats_and_no_invented_dates():
    assert parse_headlines('10 october 1801 -- “STOCKS NEWS US-Banks reacting differently to government plan”')[0] == {
        'date':'1801-10-10','headline':'STOCKS NEWS US-Banks reacting differently to government plan','evidence':'manual_import'}
    assert parse_headlines('date,headline,ignored\n2025-10-10,"A headline, with comma",future_price')[0]['headline']=='A headline, with comma'
    assert normalize_headline('2025-10-03T01:00:00Z','Headline')['date']=='2025-10-02'
    for text in ['October 2025 -- Incomplete date','10 October -- Missing year','2025-10-10 headline without separator']:
        with pytest.raises(ValueError): parse_headlines(text)


def test_calendar_windows_weekends_dedup_target_cutoff_and_dates():
    rows = headlines()
    counts = [len(packet(rows,'BABA','2025-10-17',w)['headlines']) for w in [5,10,20]]
    assert counts == [2,2,3]
    result=packet(rows+rows,'BABA','2025-10-17',20)
    assert len(result['headlines'])==3
    assert 'Target day headline' not in result['text'] and '2025-10-12 -- Weekend partnership' in result['text']
    assert 'source' not in result['text'] and 'fixture-secret' not in result['text']
    assert result['cutoff']=='2025-10-17T00:00:00-04:00'
    assert packet(rows,'BABA','2025-11-07',5)['cutoff'].endswith('-05:00')
    assert packet(rows,'BABA','2025-11-28',5)['close_at'].startswith('2025-11-28T18:00')
    with pytest.raises(ValueError): packet(rows,'BABA','2025-10-18',5)
    with pytest.raises(ValueError): packet(rows,'BABA','2025-12-25',5)
    assert trading_dates(date(2025,10,17),date(2025,10,20))==['2025-10-17','2025-10-20']


def test_strict_availability_rejects_backfilled_and_date_only():
    rows=[normalize_headline('2025-10-16','One',available_at='2026-01-01T00:00:00Z',timestamp='2025-10-16T10:00:00Z'),
          normalize_headline('2025-10-16','Two'),
          normalize_headline('2025-10-16','Three',available_at='2025-10-16T11:00:00Z',timestamp='2025-10-16T10:00:00Z')]
    p=packet(rows,'BABA','2025-10-17',5,strict=True)
    assert [r['headline'] for r in p['headlines']]==['Three'] and len(p['exclusions'])==2


def test_google_request_is_one_turn_allowlisted_and_rejects_tools():
    seen=[]
    def handler(req):
        seen.append(req)
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[
            {'text':'{"direction":"UP","reason":"Fixture evidence."}'}]}}], 'modelVersion':'fixture-v1'})
    client=GemmaClient('test-private-key',transport=httpx.MockTransport(handler))
    data=client.predict('Ticker: BABA\n2025-10-16 -- Headline')
    body=json.loads(seen[0].content)
    assert set(body)=={'systemInstruction','contents','generationConfig'}
    assert len(body['contents'])==1 and body['contents'][0]['role']=='user'
    assert body['generationConfig']['thinkingConfig']['thinkingLevel']=='minimal'
    assert 'test-private-key' not in seen[0].url.query.decode()
    assert 'test-private-key' not in json.dumps(body)
    assert data['model_version']=='fixture-v1' and parse_prediction(data['raw_text'])['direction']=='UP'
    client.close()
    for extra in [{'groundingMetadata':{'webSearchQueries':['BABA outcome']}},
                  {'content':{'parts':[{'functionCall':{'name':'search'}}]}}]:
        client=GemmaClient('secret',transport=httpx.MockTransport(lambda _:httpx.Response(200,json={'candidates':[extra]})))
        with pytest.raises(ProviderError):client.predict('headline')
        client.close()


def test_provider_errors_are_redacted_and_prediction_schema_strict():
    client=GemmaClient('PRIVATE',transport=httpx.MockTransport(lambda _:httpx.Response(403,json={'error':'PRIVATE'})))
    with pytest.raises(ProviderError) as e:client.check()
    assert 'PRIVATE' not in str(e.value)
    client.close()
    for text in ['UP', '{"direction":"UNKNOWN"}', '["UP"]']:
        with pytest.raises(ValueError):parse_prediction(text)


class FixtureClient:
    prompts=[]
    fail=False
    def __init__(self,key): assert key=='fixture-secret-only'
    def check(self):return {'name':'models/'+MODEL,'version':'fixture-v1'}
    def count(self,text):return 20
    def close(self):pass
    def predict(self,text):
        self.prompts.append(text)
        if self.fail:raise ProviderError('Quota reached',429)
        return {'raw_text':'{"direction":"UP","reason":"Synthetic fixture only."}',
                'model_version':'fixture-v1','usage':{'promptTokenCount':20}}


def test_worker_freezes_inputs_labels_separate_resumes_and_exports(exp):
    FixtureClient.prompts=[]; FixtureClient.fail=False
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5,10,20])
    save_labels(exp,key,'date,open,close\n2025-10-17,100,102\n')
    work(exp,key,client_factory=FixtureClient,sleep=lambda _:None)
    assert exp.study(key)['status']=='completed'
    assert len(FixtureClient.prompts)==3
    assert all('102' not in p and '100' not in p and 'FixtureClient' not in p for p in FixtureClient.prompts)
    assert all(row['accuracy']==1 for row in report(exp,key)['summary'])
    payload=bundle(exp,key)
    assert b'fixture-secret-only' not in payload
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name,expected in manifest['files_sha256'].items():assert digest(z.read(name))==expected
        assert 'fixture-secret-only' not in ''.join(z.read(n).decode() for n in z.namelist())
    # Reopening the worker cannot resend already completed predictions.
    work(exp,key,client_factory=FixtureClient,sleep=lambda _:None)
    assert len(FixtureClient.prompts)==3


def test_quota_failure_keeps_completed_results_and_raw_attempts(exp):
    FixtureClient.fail=True; FixtureClient.prompts=[]
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5,10])
    work(exp,key,client_factory=FixtureClient,sleep=lambda _:None)
    assert exp.study(key)['status']=='paused'
    assert exp.rows('SELECT status FROM predictions ORDER BY id')[0]['status']=='failed'
    assert exp.rows('SELECT * FROM attempts WHERE stage="error"')
    FixtureClient.fail=False
    exp.execute("UPDATE predictions SET status='pending' WHERE status='failed'")
    exp.execute("UPDATE studies SET status='starting' WHERE id=?",(key,))
    work(exp,key,client_factory=FixtureClient,sleep=lambda _:None)
    assert exp.study(key)['status']=='completed'
    assert len(exp.rows('SELECT * FROM predictions WHERE status="completed"'))==2


def test_pause_and_resume_preserves_completed_call(exp):
    calls=[]
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5,10])
    class PausingClient(FixtureClient):
        def predict(self,text):
            calls.append(text)
            if len(calls)==1:pause(exp,key)
            return super().predict(text)
    FixtureClient.fail=False
    work(exp,key,client_factory=PausingClient,sleep=lambda _:None)
    assert exp.study(key)['status']=='paused'
    assert [r['status'] for r in exp.rows('SELECT status FROM predictions ORDER BY id')]==['completed','pending']
    exp.execute("UPDATE studies SET status='starting' WHERE id=?",(key,))
    work(exp,key,client_factory=PausingClient,sleep=lambda _:None)
    assert exp.study(key)['status']=='completed' and len(calls)==2


def test_pause_while_worker_starting_makes_no_requests(exp):
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5])
    pause(exp,key)
    work(exp,key,client_factory=lambda _:pytest.fail('Paused worker must not contact Google'))
    assert exp.study(key)['status']=='paused'
    assert exp.rows('SELECT status FROM predictions')[0]['status']=='pending'


def test_collector_to_packet_without_bodies_or_audit_metadata(store,job,company):
    from newscollector.core import Candidate,timestamp
    from newscollector.worker import discover
    from newscollector.experiments import collector_headlines
    c=Candidate(url='https://example.org/company-news',title='Test Robotics announces a factory',
                publisher='Fixture publisher',source='feeds',published=timestamp('2026-10-22T23:00:00Z'),
                metadata={'forbidden_label':'UP','description':'Not sent to model'})
    discover(store,job,1,'TST',company,c)
    records,notes=collector_headlines(store,'TST')
    p=packet(records,'TST','2026-10-23',5)
    assert '2026-10-22 -- Test Robotics announces a factory' in p['text']
    assert 'forbidden_label' not in p['text'] and 'Fixture publisher' not in p['text']
    assert len(p['headlines'])==1


def test_shared_date_comparison_and_flat_denominators(exp):
    key=create_study(exp,'BABA',headlines(),['2025-10-17','2025-10-20'],[5,20])
    exp.execute("UPDATE predictions SET status='completed',direction='UP' WHERE status='pending'")
    # Explicit failure affects only the shared-date comparison, without removing it from planned counts.
    exp.execute("UPDATE predictions SET status='failed',direction=NULL WHERE target='2025-10-20' AND window=5")
    save_labels(exp,key,'date,open,close\n2025-10-17,100,102\n2025-10-20,100,98')
    r=report(exp,key)
    assert [s['scored_dates'] for s in r['summary']]==[1,2]
    assert [s['shared_dates'] for s in r['summary']]==[1,1]
    assert [s['shared_accuracy'] for s in r['summary']]==[1,1]
    save_labels(exp,key,'date,open,close\n2025-10-17,100,100\n2025-10-20,100,98')
    r=report(exp,key)
    assert r['label_balance']['FLAT']==1 and r['summary'][0]['scored_dates']==0


def test_repeats_not_extra_dates_and_labels_validation(exp):
    FixtureClient.fail=False
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5],repeats=3)
    work(exp,key,client_factory=FixtureClient,sleep=lambda _:None)
    save_labels(exp,key,'date,open,close\n2025-10-17,100,98\n')
    r=report(exp,key)
    assert r['summary'][0]['scored_dates']==1 and r['summary'][0]['accuracy']==0
    assert r['repeat_agreement']==1 and len(r['rows'])==3
    for text in ['date,open,close\n2025-10-18,1,2', 'date,open,close\n2025-10-17,nan,2',
                 'date,open,close\n2025-10-17,1,2\n2025-10-17,2,3']:
        with pytest.raises(ValueError):labels_from_csv(text,'BABA')
    assert labels_from_csv('date,open,close\n2025-10-17,100,100','BABA')['2025-10-17']['direction']=='FLAT'
    assert block_interval([1]*20)==[1,1] and block_interval([1]*19) is None
    with pytest.raises(ValueError,match='none of this experiment'):
        save_labels(exp,key,'date,open,close\n2025-10-16,100,101')


def test_new_experiment_ui_empty_and_manual_preview(tmp_path,monkeypatch):
    monkeypatch.setenv('NEWS_COLLECTOR_HOME',str(tmp_path/'ui'))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/newscollector/dashboard.py'),default_timeout=40).run()
    app.sidebar.radio[0].set_value('Predict').run()
    assert not app.exception
    assert any('No saved headlines yet' in x.value for x in app.info)
    next(x for x in app.radio if x.label=='Headline input').set_value('Paste / upload headlines').run()
    next(x for x in app.text_area if x.label=='Paste dated headlines').set_value('2025-10-16 -- Company announces a partnership').run()
    next(x for x in app.date_input if x.label=='Prediction date').set_value(date(2025,10,17)).run()
    assert not app.exception
    text=next(x for x in app.text_area if x.label=='Exact dated-headline input').value
    assert '2025-10-16 -- Company announces a partnership' in text
    assert next(x for x in app.button if x.label=='Predict UP / DOWN').disabled
    app.sidebar.radio[0].set_value('History').run()
    assert not app.exception and any('No predictions saved yet' in x.value for x in app.info)


def test_saved_prediction_report_ui_and_separate_prices(exp,monkeypatch):
    key=create_study(exp,'BABA',headlines(),['2025-10-17'],[5])
    exp.execute("UPDATE predictions SET status='completed',direction='UP',reason='Fixture only' WHERE study_id=?",(key,))
    exp.execute("UPDATE studies SET status='completed' WHERE id=?",(key,))
    monkeypatch.setenv('NEWS_COLLECTOR_HOME',str(exp.home))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/newscollector/dashboard.py'),default_timeout=40).run()
    app.sidebar.radio[0].set_value('History').run()
    assert not app.exception
    next(x for x in app.number_input if x.label=='Actual opening price').set_value(100)
    next(x for x in app.number_input if x.label=='Actual closing price').set_value(102)
    next(x for x in app.button if x.label=='Save actual prices').click().run()
    assert not app.exception and report(exp,key)['summary'][0]['accuracy']==1
