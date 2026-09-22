"""Authenticated, retry-safe raw recording collection; never trains or deploys a model."""
import hmac
import json
import math
import os
from pathlib import Path
from threading import Lock
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix='/recordings')
ROOT = Path(__file__).resolve().parents[2] / 'data'
FIELDS = {'timestamps': 0, 'accel': 3, 'gyro': 3, 'orient': 3, 'gnss': 6, 'gnss_timestamps': 0}
lock = Lock()


def authorize(authorization: str = Header(default='')):
    token = os.environ.get('NAVIGATORS_SYNC_TOKEN', '')
    if not token or not hmac.compare_digest(authorization.encode(), ('Bearer ' + token).encode()):
        raise HTTPException(401, 'Enter the pairing code printed by the PC server.')


@router.get('/status', dependencies=[Depends(authorize)])
def status():
    return {'status': 'connected', 'completed_trips': len(list((ROOT / 'phone_recordings').glob('*.json'))),
            'storage': 'data/phone_recordings', 'training': 'Not automatic; separate walking and vehicle datasets.'}


def store_chunk(trip_id, sequence, payload):
    data = payload.get('data', {})
    count = len(data.get('timestamps', []))
    if not 0 <= sequence < 720 or count > 1000 or not isinstance(payload.get('final'), bool):
        raise HTTPException(422, 'Invalid recording batch size or sequence.')
    if payload.get('metadata', {}).get('navigation_mode') not in ('walking', 'vehicle'):
        raise HTTPException(422, 'Recording must identify walking or vehicle mode.')
    for field, width in FIELDS.items():
        values = data.get(field)
        if not isinstance(values, list) or len(values) != count:
            raise HTTPException(422, 'Recording arrays must have equal lengths.')
        for row in values:
            if width and (not isinstance(row, list) or len(row) != width):
                raise HTTPException(422, 'Invalid sensor row.')
            for value in row if width else [row]:
                if value is None and field in ('gnss', 'gnss_timestamps', 'orient'):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise HTTPException(422, 'Invalid sensor value.')
    times = data['timestamps']
    if any(b <= a for a, b in zip(times, times[1:])):
        raise HTTPException(422, 'Timestamps must increase.')
    folder = ROOT / 'phone_recordings' / 'partial' / str(trip_id)
    destination = ROOT / 'phone_recordings' / f'trip_{trip_id}.json'
    encoded = json.dumps(payload, allow_nan=False, separators=(',', ':'))
    with lock:
        folder.mkdir(parents=True, exist_ok=True)
        chunk = folder / f'{sequence:04d}.json'
        if chunk.exists():
            if chunk.read_text() != encoded:
                raise HTTPException(409, 'This batch already exists with different data.')
        else:
            if destination.exists():
                raise HTTPException(409, 'Recording is already finalized.')
            if sequence:
                previous = folder / f'{sequence - 1:04d}.json'
                if not previous.exists():
                    raise HTTPException(409, 'Send earlier batches first.')
                prior = json.loads(previous.read_text())
                if prior['metadata'] != payload['metadata'] or (times and prior['data']['timestamps'] and times[0] <= prior['data']['timestamps'][-1]):
                    raise HTTPException(422, 'Recording identity or timestamp discontinuity.')
            temporary = chunk.with_suffix('.tmp')
            temporary.write_text(encoded)
            temporary.replace(chunk)
        if payload['final'] and not destination.exists():
            merged = {key: [] for key in FIELDS}
            for index in range(sequence + 1):
                part = json.loads((folder / f'{index:04d}.json').read_text())
                for key in FIELDS:
                    merged[key].extend(part['data'][key])
            metadata = {**payload['metadata'], 'num_frames': len(merged['timestamps'])}
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps({'metadata': metadata, 'data': merged}, allow_nan=False))
            temporary.replace(destination)
    return {'stored': sequence, 'complete': payload['final'], 'frames': count}


@router.post('/{trip_id}/{sequence}', dependencies=[Depends(authorize)])
async def receive(trip_id: UUID, sequence: int, request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 512 * 1024:
            raise HTTPException(413, 'Batch exceeds 512 KB.')
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), dict) or not isinstance(payload.get('metadata'), dict):
            raise ValueError()
        res = await run_in_threadpool(store_chunk, trip_id, sequence, payload)
        

        if res.get("complete"):
            import asyncio
            from .server import ws_manager
            asyncio.create_task(ws_manager.broadcast_log('SUCCESS', f'Trip {trip_id} fully synced!'))
        return res
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, 'Invalid recording JSON.') from error
