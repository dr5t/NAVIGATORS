"""Raw-upload transport tests; fixtures are not training or accuracy evidence."""
import asyncio
import json
from uuid import uuid4

from fastapi import FastAPI
import pytest
from src.api import recordings


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    monkeypatch.setattr(recordings, 'ROOT', tmp_path)
    monkeypatch.setenv('NAVIGATORS_SYNC_TOKEN', 'test-pairing-code')
    app = FastAPI()
    app.include_router(recordings.router)
    return app, tmp_path


def request(app, path, payload, token='test-pairing-code'):
    messages = []
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(payload).encode(), 'more_body': False}
    async def send(message):
        messages.append(message)
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'method': 'POST', 'path': path, 'query_string': b'',
             'scheme': 'https', 'http_version': '1.1', 'server': ('localhost', 8443), 'client': ('127.0.0.1', 1234),
             'headers': [(b'authorization', ('Bearer ' + token).encode()), (b'content-type', b'application/json')]}
    asyncio.run(app(scope, receive, send))
    return next(message['status'] for message in messages if message['type'] == 'http.response.start')


def batch(start=0, final=False):
    return {'metadata': {'navigation_mode': 'walking', 'provenance': 'synthetic transport fixture'}, 'final': final,
            'data': {'timestamps': [start, start + 0.02], 'accel': [[0, 0, 9.81]] * 2,
                     'gyro': [[0, 0, 0]] * 2, 'orient': [[None, None, None]] * 2,
                     'gnss': [[None] * 6] * 2, 'gnss_timestamps': [None, None]}}


def test_pairing_order_retries_and_final_recording(receiver):
    app, folder = receiver
    trip = uuid4()
    base = f'/recordings/{trip}'
    assert request(app, base + '/0', batch(), token='wrong') == 401
    assert not (folder / 'phone_recordings').exists()
    assert request(app, base + '/0', batch()) == 200
    assert request(app, base + '/0', batch()) == 200
    assert request(app, base + '/2', batch(2, final=True)) == 409
    assert request(app, base + '/1', batch(1, final=True)) == 200
    assert request(app, base + '/1', batch(1, final=True)) == 200
    assert request(app, base + '/1', batch(3, final=True)) == 409
    saved = json.loads((folder / 'phone_recordings' / f'trip_{trip}.json').read_text())
    assert saved['data']['timestamps'] == [0, 0.02, 1, 1.02]
    assert saved['data']['gnss'] == [[None] * 6] * 4
    assert saved['metadata']['navigation_mode'] == 'walking'
    assert saved['metadata']['num_frames'] == 4


def test_malformed_and_oversized_uploads_are_rejected(receiver):
    app, _ = receiver
    path = f'/recordings/{uuid4()}/0'
    invalid = batch()
    invalid['data']['timestamps'] = [1, 1]
    assert request(app, path, invalid) == 422
    invalid = batch()
    invalid['data']['gyro'][0] = [0, 'bad', 0]
    assert request(app, path, invalid) == 422
    assert request(app, path, {'padding': 'x' * (513 * 1024)}) == 413
    assert request(app, '/recordings/not-a-uuid/0', batch()) == 422
