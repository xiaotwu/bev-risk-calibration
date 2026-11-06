"""Recover the six source recording clocks from the official merged NGSIM table."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import datetime as dt
import gzip
import hashlib
import io
import json

from acquire_ngsim import API, CATALOG, DOI, ROOT, download, get, query, sha, write_json


SOURCE = ROOT / 'raw' / 'receipt.json'
DESTINATION = ROOT / 'raw_recordings'
PERIOD_MS = 100
ORIGIN_EXPRESSION = 'global_time-100*frame_id'
IDS = {'i-80': ['i80_1600', 'i80_1700', 'i80_1715'],
       'us-101': ['us101_0750', 'us101_0805', 'us101_0820']}
ORIGINS = {'i-80': [1113433134900, 1113436766900, 1113437564900],
           'us-101': [1118846978900, 1118847842000, 1118848757000]}


def catalog(site):
    url = query(select=f'{ORIGIN_EXPRESSION} as origin,count(*) as n,'
                       'min(global_time) as first_ms,max(global_time) as last_ms',
                where=f"location='{site}'", group=ORIGIN_EXPRESSION, order='origin', limit=20)
    with get(url) as response:
        rows = [{key: int(value) for key, value in row.items()}
                for row in csv.DictReader(io.TextIOWrapper(response, encoding='utf-8'))]
    if [row['origin'] for row in rows] != ORIGINS[site]:
        raise ValueError(f'The independently enumerated recording clocks changed for {site}: {rows}')
    return dict(url=url, rows=rows)


def margins(source_blocks, origin_rows):
    """Ranges with active recording clocks that the nominal downloads left out."""
    boundary = sorted({edge for start, end in source_blocks + origin_rows for edge in (start, end)})
    result = []
    for start, end in zip(boundary, boundary[1:]):
        needed = any(left <= start < right for left, right in origin_rows)
        covered = any(left <= start < right for left, right in source_blocks)
        if needed and not covered:
            if result and start == result[-1][1]:
                result[-1] = result[-1][0], end
            else:
                result.append((start, end))
    return result


def run():
    DESTINATION.mkdir(exist_ok=False)
    (DESTINATION / 'shards').mkdir()
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    if original['status'] != 'complete' or original['nominal_minutes_per_block'] != 15:
        raise ValueError('Complete 15-minute nominal windows required')
    with ThreadPoolExecutor(max_workers=6) as pool:
        catalogs = dict(zip(IDS, pool.map(catalog, IDS)))
        todo = []
        for site, cat in catalogs.items():
            initial = [(row['start_ms'], row['end_ms']) for row in original['blocks'] if row['site'] == site]
            observed = [(row['first_ms'], row['last_ms'] + PERIOD_MS) for row in cat['rows']]
            todo.extend((site, start, end) for start, end in margins(initial, observed))
        added = [(site, pool.submit(download, DESTINATION, f'{site}_margin', site, start, end))
                 for site, start, end in todo]
        extras = [{**future.result(), 'site': site} for site, future in added]
    inputs = []
    for row in original['blocks']:
        inputs.append({key: row[key] for key in ('site', 'path', 'sha256', 'rows', 'start_ms', 'end_ms')})
    for row in extras:
        inputs.append({**row, 'path': (DESTINATION / row['path']).relative_to(ROOT).as_posix()})
    expected = sum(row['n'] for cat in catalogs.values() for row in cat['rows'])
    if sum(row['rows'] for row in inputs) != expected:
        raise ValueError('Original and margin counts do not exhaust the exact source clock counts')
    blocks = []
    for site, cat in catalogs.items():
        stats, streams = {}, {}
        with ExitStack() as stack:
            for block_id, row in zip(IDS[site], cat['rows']):
                origin = row['origin']
                path = DESTINATION / f'{block_id}.csv.gz'
                stats[origin] = dict(id=block_id, path=path, original=row, rows=0, first=None, last=None,
                                     h=hashlib.sha256(), duplicates=0, frame_vehicles=set())
                raw = stack.enter_context(path.open('xb'))
                streams[origin] = stack.enter_context(gzip.GzipFile(fileobj=raw, filename='', mode='wb',
                                                                   compresslevel=3, mtime=0))
            header = None
            site_inputs = sorted((row for row in inputs if row['site'] == site), key=lambda row: row['start_ms'])
            for row in site_inputs:
                path = ROOT / row['path']
                if sha(path) != row['sha256']:
                    raise ValueError(f'Input hash mismatch: {path}')
                with gzip.open(path, 'rb') as stream:
                    supplied = stream.readline()
                    if header is None:
                        header = supplied
                        for origin, destination in streams.items():
                            destination.write(header)
                            stats[origin]['h'].update(header)
                    if supplied != header:
                        raise ValueError(f'Input header mismatch: {path}')
                    observed = 0
                    for line in stream:
                        fields = line.split(b',')
                        vehicle, timestamp, frame = (int(fields[index].strip(b'"')) for index in (0, 1, 8))
                        origin = timestamp - PERIOD_MS * frame
                        if origin not in streams:
                            raise ValueError(f'Unexpected source recording clock {origin} in {path}')
                        stat = stats[origin]
                        if stat['last'] is not None and timestamp < stat['last']:
                            raise ValueError(f'Non-monotone timestamps for {stat["id"]}')
                        if stat['last'] != timestamp:
                            stat['frame_vehicles'].clear()
                        elif vehicle in stat['frame_vehicles']:
                            stat['duplicates'] += 1
                        stat['frame_vehicles'].add(vehicle)
                        stat['first'] = timestamp if stat['first'] is None else stat['first']
                        stat['last'] = timestamp
                        stat['rows'] += 1
                        stat['h'].update(line)
                        streams[origin].write(line)
                        observed += 1
                if observed != row['rows']:
                    raise ValueError(f'Input row count mismatch: {path}')
                print(f'{site} {path.name}: {observed} source rows assigned', flush=True)
        for origin, stat in stats.items():
            expected_origin = stat['original']
            if (stat['rows'], stat['first'], stat['last']) != (
                    expected_origin['n'], expected_origin['first_ms'], expected_origin['last_ms']):
                raise ValueError(f'Clock reconstruction differs from independent source query: {stat["id"]}')
            path = stat['path']
            h = hashlib.sha256()
            with gzip.open(path, 'rb') as stream:
                for part in iter(lambda: stream.read(1 << 20), b''):
                    h.update(part)
            if h.hexdigest() != stat['h'].hexdigest():
                raise ValueError(f'Lossless archive check failed: {path}')
            where = f"location='{site}' and {ORIGIN_EXPRESSION}={origin}"
            blocks.append(dict(id=stat['id'], site=site, origin_ms=origin,
                               path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                               decompressed_sha256=h.hexdigest(), rows=stat['rows'], bytes=path.stat().st_size,
                               first_ms=stat['first'], last_ms=stat['last'], start_ms=stat['first'],
                               end_ms=stat['last'] + PERIOD_MS, duplicate_vehicle_frame_rows=stat['duplicates'],
                               url=query(select=','.join(original['columns']), where=where),
                               url_role='source recording selection; actual downloads use input shard URLs'))
    write_json(DESTINATION / 'receipt.json', dict(
        status='complete', source='FHWA NGSIM, USDOT ITS DataHub', api=API, catalog=CATALOG, doi=DOI,
        origin_invariant='global_time_ms - 100 * frame_id equals the exact recording clock origin',
        origin_catalogs=catalogs, metadata=original['metadata'], columns=original['columns'],
        original_nominal_receipt=dict(path=SOURCE.relative_to(ROOT).as_posix(), sha256=sha(SOURCE)),
        inputs=inputs, margin_rows=sum(row['rows'] for row in extras), total_rows=expected, blocks=blocks,
        complete_site_table_rows=True, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    run()
