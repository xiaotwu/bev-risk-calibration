"""Fetch six nominal freeway time windows from the official FHWA NGSIM API."""

import argparse
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
API = 'https://data.transportation.gov/resource/8ect-6jqj.csv'
DOI = 'https://doi.org/10.21949/1504477'
CATALOG = ('https://data.transportation.gov/Automobiles/'
           'Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/data')
I80_METADATA = ('https://data.transportation.gov/api/views/8ect-6jqj/files/'
                '08ab2236-3687-4904-a2b4-b78c7ce42f36?download=true')
US101_METADATA = ('https://data.transportation.gov/api/views/8ect-6jqj/files/'
                  'ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true')
COLUMNS = ('vehicle_id', 'global_time', 'preceding', 'lane_id', 'local_y', 'v_length', 'v_class',
           'v_vel', 'frame_id', 'total_frames', 'local_x', 'space_headway')
BLOCKS = (
    ('i80_1600', 'i-80', '2005-04-13T23:00:00Z'),
    ('i80_1700', 'i-80', '2005-04-14T00:00:00Z'),
    ('i80_1715', 'i-80', '2005-04-14T00:15:00Z'),
    ('us101_0750', 'us-101', '2005-06-15T14:50:00Z'),
    ('us101_0805', 'us-101', '2005-06-15T15:05:00Z'),
    ('us101_0820', 'us-101', '2005-06-15T15:20:00Z'),
)
LIMIT = 500_000


def millis(instant):
    return int(dt.datetime.fromisoformat(instant.replace('Z', '+00:00')).timestamp() * 1000)


def query(**params):
    return API + '?' + urlencode({f'${key}': value for key, value in params.items()})


def condition(site, start, end):
    return f"location='{site}' and global_time >= {start} and global_time < {end}"


def data_url(where):
    return query(select=','.join(COLUMNS), where=where,
                 order='global_time,vehicle_id,frame_id,total_frames', limit=LIMIT)


def get(url):
    return urlopen(Request(url, headers={'User-Agent': 'LayerGate-BEV reproducible research/1'}), timeout=65)


def count(url):
    with get(url) as response:
        result = list(csv.DictReader(io.TextIOWrapper(response, encoding='utf-8')))
    if len(result) != 1:
        raise ValueError(f'Unexpected count response for {url}')
    return int(result[0]['n'])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def download(output, block_id, site, start, end):
    where = condition(site, start, end)
    url = data_url(where)
    count_url = query(select='count(*) as n', where=where)
    stem = output / 'shards' / f'{block_id}_{start}_{end}'
    destination, receipt = stem.with_suffix('.csv.gz'), stem.with_suffix('.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        if prior['url'] != url or sha(destination) != prior['sha256']:
            raise ValueError(f'Retained shard differs: {stem}')
        return prior
    if destination.exists():
        raise FileExistsError(destination)
    for attempt in range(1, 5):
        temporary = stem.with_suffix(f'.attempt_{attempt}.csv.gz')
        try:
            expected = count(count_url)
            if expected >= LIMIT:
                raise ValueError(f'Shard needs subdivision: {expected} rows in {where}')
            rows = 0
            first, last = None, None
            with get(url) as response, temporary.open('xb') as raw, gzip.GzipFile(
                    fileobj=raw, filename='', mode='wb', compresslevel=3, mtime=0) as stream:
                header = response.readline()
                if tuple(next(csv.reader([header.decode()]))) != COLUMNS:
                    raise ValueError(f'Unexpected header {header!r}')
                stream.write(header)
                for line in response:
                    fields = next(csv.reader([line.decode()]))
                    if len(fields) != len(COLUMNS):
                        raise ValueError(f'Unexpected row width in {where}: {line!r}')
                    timestamp = int(fields[1])
                    if not start <= timestamp < end or (last is not None and timestamp < last):
                        raise ValueError(f'Unexpected timestamp/order in {where}: {timestamp}')
                    first = timestamp if first is None else first
                    last = timestamp
                    rows += 1
                    stream.write(line)
            if rows != expected:
                raise IOError(f'Incomplete response: got {rows}, expected {expected}')
            temporary.rename(destination)
            proof = dict(path=destination.relative_to(output).as_posix(), start_ms=start, end_ms=end,
                         rows=rows, first_ms=first, last_ms=last, url=url, count_url=count_url,
                         sha256=sha(destination), bytes=destination.stat().st_size, attempts=attempt)
            write_json(receipt, proof)
            print(f'{block_id} {start} {rows} rows verified', flush=True)
            return proof
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, FileExistsError) or attempt == 4:
                raise
            print(f'{block_id} {start} attempt {attempt}: {exc}', flush=True)
            time.sleep(attempt * 3)
    raise RuntimeError('Unreachable')


def assemble(output, block, shards):
    block_id, site, start, end = block
    path = output / f'{block_id}.csv.gz'
    with path.open('xb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb',
                                              compresslevel=3, mtime=0) as stream:
        for index, shard in enumerate(shards):
            with gzip.open(output / shard['path'], 'rb') as source:
                header = source.readline()
                if not index:
                    stream.write(header)
                for data in iter(lambda: source.read(1 << 20), b''):
                    stream.write(data)
    where = condition(site, start, end)
    return dict(id=block_id, site=site, path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                rows=sum(shard['rows'] for shard in shards), start_ms=start, end_ms=end,
                url=query(select=','.join(COLUMNS), where=where),
                url_role='canonical filter; use the bounded shard URLs for full downloads',
                first_ms=min(shard['first_ms'] for shard in shards if shard['first_ms'] is not None),
                last_ms=max(shard['last_ms'] for shard in shards if shard['last_ms'] is not None),
                bytes=path.stat().st_size, shards=shards)


def main(workers, minutes):
    output = ROOT / ('raw' if minutes == 15 else 'raw_5min_fallback')
    (output / 'shards').mkdir(parents=True, exist_ok=True)
    if (output / 'receipt.json').exists():
        raise FileExistsError(output / 'receipt.json')
    blocks = [(block_id, site, millis(utc), millis(utc) + 60_000 * minutes)
              for block_id, site, utc in BLOCKS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {block_id: [pool.submit(download, output, block_id, site, t, min(t + 120_000, end))
                              for t in range(start, end, 120_000)]
                   for block_id, site, start, end in blocks}
        proofs = {block_id: [future.result() for future in batch]
                  for block_id, batch in futures.items()}
    receipts = [assemble(output, block, proofs[block[0]]) for block in blocks]
    write_json(output / 'receipt.json', dict(
        status='complete', source='FHWA NGSIM, USDOT ITS DataHub', catalog=CATALOG, doi=DOI,
        metadata={'i-80': I80_METADATA, 'us-101': US101_METADATA}, columns=COLUMNS,
        nominal_minutes_per_block=minutes, timezone='America/Los_Angeles (PDT, UTC-07:00)',
        interpretation=('Combined official API time slices within the six nominal recording intervals. '
                        'The merged table does not identify original source files; some recordings contain '
                        'boundary lead-in or tail rows outside those intervals. Nominal blocks are '
                        'analysis boundaries, not independently sampled recordings or inferred original-file membership.'),
        total_rows=sum(block['rows'] for block in receipts), blocks=receipts,
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--minutes', type=int, choices=(5, 15), default=15)
    args = parser.parse_args()
    main(args.workers, args.minutes)
