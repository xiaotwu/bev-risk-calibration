"""Archive the ten official OpenACC AstaZero platoon CSVs losslessly."""

from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'openacc' / 'raw'
BASE = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/AstaZero/'
DOI = 'https://doi.org/10.2905/JRC.KMH3D00'
RUNS = tuple(f'ASta_040719_platoon{i}' for i in range(3, 11)) + (
    'ASta_050719_platoon1', 'ASta_050719_platoon2')
SUPPORT = ('AstaZero_notes.txt', 'AstaZero_veh_specifications.csv', 'copyright.txt')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def get(name):
    url = BASE + name
    return urlopen(Request(url, headers={'User-Agent': 'LayerGate-BEV reproducible research/1'}), timeout=75)


def archive_support(name):
    path = OUTPUT / name
    with get(name) as response:
        data = response.read()
        if response.status != 200 or not data:
            raise ValueError(f'Unexpected supporting material response: {name}')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Retained supporting material differs: {name}')
    else:
        with path.open('xb') as stream:
            stream.write(data)
    return dict(path=path.relative_to(ROOT).as_posix(), url=BASE + name,
                sha256=sha(path), bytes=path.stat().st_size)


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def archive_run(run):
    path = OUTPUT / f'{run}.csv.gz'
    proof = OUTPUT / f'{run}.json'
    name = run + '.csv'
    if proof.exists():
        existing = json.loads(proof.read_text(encoding='utf-8'))
        if existing['url'] != BASE + name or sha(path) != existing['sha256']:
            raise ValueError(f'Retained run differs: {run}')
        return existing
    if path.exists():
        raise FileExistsError(path)
    for attempt in range(1, 5):
        temporary = OUTPUT / f'{run}.attempt_{attempt}.csv.gz'
        try:
            count, received = 0, 0
            h = hashlib.sha256()
            with get(name) as response, temporary.open('xb') as raw, gzip.GzipFile(
                    fileobj=raw, filename='', mode='wb', compresslevel=3, mtime=0) as stream:
                if response.status != 200:
                    raise OSError(f'Expected full source file, got HTTP {response.status}: {run}')
                expected = response.headers.get('Content-Length')
                last_modified = response.headers.get('Last-Modified')
                for line in response:
                    h.update(line)
                    received += len(line)
                    count += 1
                    stream.write(line)
            if received < 1_000_000 or (expected is not None and received != int(expected)):
                raise OSError(f'Incomplete response: received {received}, expected {expected} for {run}')
            with gzip.open(temporary, 'rt', encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                metadata = [next(reader)[0] for _ in range(5)]
                header = next(reader)
                if metadata != ['Date', 'Vehicle_order', 'Number_of_vehicles', 'ACC', 'Distance_setting']:
                    raise ValueError(f'Unexpected metadata prefix for {run}: {metadata}')
                if not {'Time', 'Speed1', 'Speed2', 'IVS1'}.issubset(header):
                    raise ValueError(f'Unexpected header for {run}: {header}')
                rows = sum(bool(row and any(row)) for row in reader)
            temporary.rename(path)
            result = dict(id=run, path=path.relative_to(ROOT).as_posix(), url=BASE + name,
                          rows=rows, lines=count, bytes=path.stat().st_size, sha256=sha(path),
                          original_bytes=received, original_sha256=h.hexdigest(), last_modified=last_modified,
                          header=header, attempts=attempt)
            write_json(proof, result)
            print(f'{run} {rows} rows archived', flush=True)
            return result
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, FileExistsError) or attempt == 4:
                raise
            print(f'{run} attempt {attempt}: {exc}', flush=True)
            time.sleep(attempt * 3)
    raise RuntimeError('Unreachable')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    receipt = OUTPUT / 'receipt.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    with ThreadPoolExecutor(max_workers=2) as pool:
        support = list(pool.map(archive_support, SUPPORT))
        runs = list(pool.map(archive_run, RUNS))
    write_json(receipt, dict(status='complete', source='European Commission Joint Research Centre OpenACC',
                             campaign='AstaZero', index_url=BASE, doi=DOI,
                             license='CC BY 4.0, see the archived provider copyright notice',
                             adaptation='Original bytes stored in gzip members without tabular changes',
                             units={'Time': 's', 'Speed1': 'm/s', 'Speed2': 'm/s', 'IVS1': 'm, bumper to bumper'},
                             support=support, runs=runs, total_rows=sum(run['rows'] for run in runs),
                             completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
