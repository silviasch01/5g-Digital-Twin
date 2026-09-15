#!/usr/bin/env python3
"""
generate_npz_datasets.py
Genera i file .npz da dataset_v3 e dataset_real_v3.

Varianti:
  activity  — usa H_activity_v*.csv per tutte le attività  (H_d già pulita)
  combined  — usa H_combined_v*.csv per le attività non-IDLE,
               H_activity_v*.csv per IDLE  (= H_s + H_d grezzo)
  both      — genera entrambe (default)

Struttura attesa:
  dataset_v3/           → subdirectory rxN/ per ogni attività  (SIM, multi-rx)
  dataset_real_v3/      → CSV direttamente nella dir attività  (REAL, single-rx)

Uso:
    python generate_npz_datasets.py
    python generate_npz_datasets.py --variants activity
    python generate_npz_datasets.py --variants combined
    python generate_npz_datasets.py --variants both --rebuild
    python generate_npz_datasets.py --dataset dataset_v3 --variants combined
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Helpers verbosity ─────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    return f'[{"█" * filled}{"░" * (width - filled)}] {done}/{total}'


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f'{m}m{s:02d}s' if m else f'{s}s'


def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return '---'
    remaining = elapsed / done * (total - done)
    return _fmt(remaining)


def _spinning_save(label: str, fn):
    """Esegue fn() in un thread e stampa uno spinner con elapsed finché non finisce."""
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    done   = threading.Event()
    t0     = time.time()

    def _spin():
        i = 0
        while not done.is_set():
            elapsed = time.time() - t0
            print(f'\r  {FRAMES[i % len(FRAMES)]}  {label}  {_fmt(elapsed)}...   ',
                  end='', flush=True)
            time.sleep(0.12)
            i += 1

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    try:
        fn()
    finally:
        done.set()
        spinner.join()
    return time.time() - t0

# ── Configurazione ────────────────────────────────────────────────────────────

DATASETS = [
    {
        'path':     'dataset_v3',
        'has_rx':   True,    # struttura rxN/ per ogni attività
        'rx_range': range(36),
    },
    {
        'path':     'dataset_real_v4',
        'has_rx':   False,   # CSV direttamente nella dir attività
        'rx_range': None,
    },
]

ACTIVITY_DIRS = {
    'FALL':  'fall_furnished',
    'IDLE':  'furnished_idle',
    'JUMP':  'jump_furnished',
    'RUN':   'run_furnished',
    'STAND': 'stand_furnished',
    'WALK':  'walk_furnished',
}

ALL_ACTIVITIES = list(ACTIVITY_DIRS.keys())

N_STEPS    = 40
N_SC       = 1272
N_FEATURES = 4

# ── I/O ───────────────────────────────────────────────────────────────────────

def _raw_csv_to_array(path: Path) -> np.ndarray | None:
    """Legge H_activity_v{i}.csv → (N_STEPS, N_SC, N_FEATURES) float32."""
    steps = []
    with open(path) as fh:
        fh.readline()  # header: step;0;1;...;1271
        for line in fh:
            parts = line.rstrip('\n').split(';')
            sc_vals = [list(map(float, v.split(','))) for v in parts[1:]]
            steps.append(sc_vals)
    arr = np.array(steps, dtype=np.float32)
    if arr.shape != (N_STEPS, N_SC, N_FEATURES):
        print(f'  [WARN] shape inattesa {arr.shape} — {path}', file=sys.stderr)
        return None
    return arr


def _collect_tasks(
    base_dir: Path,
    activities: list[str],
    use_combined: bool,
    has_rx: bool,
    rx_range,
) -> list[tuple[Path, str, int, str]]:
    """Raccoglie tutti i (csv_path, activity, rx_id, pattern) da leggere."""
    tasks = []
    for act in activities:
        act_dir = base_dir / ACTIVITY_DIRS[act]
        if not act_dir.exists():
            print(f'  [WARN] directory non trovata: {act_dir}', file=sys.stderr)
            continue
        pattern = ('H_combined_v*.csv'
                   if use_combined and act != 'IDLE'
                   else 'H_activity_v*.csv')
        if has_rx:
            search_dirs = [(act_dir / f'rx{rx}', rx) for rx in rx_range]
        else:
            search_dirs = [(act_dir, 0)]
        for dir_path, rx_id in search_dirs:
            if not dir_path.exists():
                continue
            for csv_path in sorted(dir_path.glob(pattern)):
                tasks.append((csv_path, act, rx_id, pattern))
    return tasks


def _load_task(args_tuple):
    """Worker: legge un singolo CSV. Restituisce (index, array, act, rx_id) o None."""
    idx, csv_path, act, rx_id = args_tuple
    arr = _raw_csv_to_array(csv_path)
    return (idx, arr, act, rx_id)


def _build_npz(
    base_dir: Path,
    activities: list[str],
    use_combined: bool,
    has_rx: bool,
    rx_range,
    out_path: Path,
    n_workers: int = 1,
) -> None:
    """Legge raw CSV in parallelo con progress live e scrive il .npz compresso."""
    from collections import Counter

    print(f'\n  Scansione file CSV...')
    tasks = _collect_tasks(base_dir, activities, use_combined, has_rx, rx_range)
    total = len(tasks)
    if total == 0:
        print(f'  [ERR] nessun file trovato — {out_path} non scritto', file=sys.stderr)
        return

    act_counts = Counter(act for _, act, _, _ in tasks)
    print(f'  File trovati: {total} totali  (workers: {n_workers})')
    for act in activities:
        if act in act_counts:
            print(f'    {act:5s}: {act_counts[act]} file')

    # indicizza i task per ricostruire l'ordine originale dopo il pool
    indexed = [(i, csv_path, act, rx_id) for i, (csv_path, act, rx_id, _) in enumerate(tasks)]

    results: list[tuple[int, np.ndarray, str, int] | None] = [None] * total
    act_done: dict[str, int] = {}
    n_errors = 0

    print()
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_task, t): t[0] for t in indexed}
        done = 0
        for fut in as_completed(futures):
            done += 1
            elapsed = time.time() - t_start
            speed   = done / elapsed if elapsed > 0 else 0
            bar     = _bar(done, total)
            eta     = _eta(elapsed, done, total)

            idx, arr, act, rx_id = fut.result()
            if arr is not None:
                results[idx] = (idx, arr, act, rx_id)
                act_done[act] = act_done.get(act, 0) + 1
            else:
                n_errors += 1

            print(
                f'\r  {bar}  {done:5d}/{total}'
                f'  {_fmt(elapsed)} el.  ETA {eta}'
                f'  {speed:5.1f} f/s   ',
                end='', flush=True,
            )

    elapsed_total = time.time() - t_start
    print(f'\r  {_bar(total, total)}  {total}/{total}'
          f'  {_fmt(elapsed_total)} totale  ({total/elapsed_total:.1f} f/s medio)' + ' ' * 10)

    if n_errors:
        print(f'  [WARN] {n_errors} file scartati per shape inattesa', file=sys.stderr)

    print(f'\n  Campioni per attività:')
    total_ok = 0
    for act in activities:
        n = act_done.get(act, 0)
        total_ok += n
        print(f'    {act:5s}: {n:5d} sample')
    print(f'    {"TOTALE":5s}: {total_ok:5d} sample')

    valid = [r for r in results if r is not None]
    if not valid:
        print(f'  [ERR] nessun dato valido — {out_path} non scritto', file=sys.stderr)
        return

    n_valid = len(valid)
    print(f'\n  Stack array ({n_valid} sample × {N_STEPS} step × {N_SC} SC × {N_FEATURES} feat)...',
          end='', flush=True)
    t_stack = time.time()
    X  = np.stack([r[1] for r in valid])
    y  = np.array([r[2] for r in valid])
    rx = np.array([r[3] for r in valid], dtype=np.int32)
    ram_mb = X.nbytes / 1024 ** 2
    print(f' fatto in {_fmt(time.time() - t_stack)}'
          f'  shape={X.shape}  RAM={ram_mb:.0f} MB  dtype={X.dtype}')

    print(f'\n  Compressione zlib (questa parte può richiedere qualche minuto)...')
    print(f'  Dati raw: {ram_mb:.0f} MB  →  target .npz compresso...')

    elapsed_save = _spinning_save(
        f'Compressione + scrittura {out_path.name}',
        lambda: np.savez_compressed(str(out_path), X=X, y=y, rx_ids=rx),
    )
    size_mb    = out_path.stat().st_size / 1024 ** 2
    ratio      = ram_mb / size_mb if size_mb > 0 else 0
    throughput = ram_mb / elapsed_save if elapsed_save > 0 else 0
    print(f'\r  ✓ Compressione completata in {_fmt(elapsed_save)}' + ' ' * 20)
    print(f'     {ram_mb:.0f} MB raw  →  {size_mb:.1f} MB su disco'
          f'  (ratio {ratio:.1f}x)  {throughput:.0f} MB/s throughput')
    print(f'\n  ✓ {out_path}  shape={X.shape}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--variants',
        choices=['activity', 'combined', 'both'],
        default='both',
        help='Variante NPZ da generare: activity, combined, both (default: both)',
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Rigenera i .npz anche se già esistenti',
    )
    parser.add_argument(
        '--dataset', default=None,
        help='Limita a un singolo dataset (es. dataset_v3 oppure dataset_real_v3)',
    )
    parser.add_argument(
        '--activities', default=','.join(ALL_ACTIVITIES),
        help=f'Attività da includere, separate da virgola (default: tutte)',
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help=f'Processi paralleli per la lettura CSV (default: {os.cpu_count()} = tutti i core)',
    )
    args = parser.parse_args()

    activities = [a.strip().upper() for a in args.activities.split(',') if a.strip()]
    invalid = [a for a in activities if a not in ACTIVITY_DIRS]
    if invalid:
        parser.error(f'Attività non valide: {invalid}  (scegli tra: {list(ACTIVITY_DIRS)})')

    to_build: list[tuple[str, bool]] = []
    if args.variants in ('activity', 'both'):
        to_build.append(('activity', False))
    if args.variants in ('combined', 'both'):
        to_build.append(('combined', True))

    base_root = Path(__file__).parent

    print(f'\n{"═" * 60}')
    print(f'  generate_npz_datasets.py')
    print(f'  Varianti  : {", ".join(s for s, _ in to_build)}')
    print(f'  Attività  : {", ".join(activities)}')
    print(f'  Rebuild   : {args.rebuild}')
    print(f'  Workers   : {args.workers} processi paralleli')
    print(f'{"═" * 60}')

    t_global = time.time()

    for ds_cfg in DATASETS:
        ds_path = base_root / ds_cfg['path']

        if args.dataset and Path(args.dataset).name != ds_path.name:
            continue

        if not ds_path.exists():
            print(f'\n[SKIP] {ds_path} non trovata')
            continue

        rx_info = (f'{len(list(ds_cfg["rx_range"]))} receiver (rx0..rx{max(ds_cfg["rx_range"])})'
                   if ds_cfg['has_rx'] else 'single-rx (flat)')
        print(f'\n{"═" * 60}')
        print(f'  Dataset : {ds_path.name}')
        print(f'  Struttura: {rx_info}')
        print(f'{"═" * 60}')

        for suffix, use_combined in to_build:
            out_npz = ds_path / f'dataset_har_{suffix}.npz'

            print(f'\n  ── Variante: {suffix} {"(H_combined per non-IDLE)" if use_combined else "(H_activity per tutte)"} ──')

            if out_npz.exists() and not args.rebuild:
                try:
                    data = np.load(str(out_npz), allow_pickle=True)
                    print(f'  [SKIP] {out_npz.name} già esistente'
                          f'  shape={data["X"].shape}'
                          f'  classi={sorted(set(data["y"].tolist()))}')
                    print(f'         (usa --rebuild per rigenerare)')
                    continue
                except Exception as exc:
                    print(f'  [WARN] {out_npz.name} corrotto ({exc.__class__.__name__}: {exc})')
                    print(f'         Rigenerazione automatica...')
                    out_npz.unlink()

            if out_npz.exists():
                print(f'  Rimozione {out_npz.name} esistente...')
                out_npz.unlink()

            t0 = time.time()
            _build_npz(
                base_dir=ds_path,
                activities=activities,
                use_combined=use_combined,
                has_rx=ds_cfg['has_rx'],
                rx_range=ds_cfg['rx_range'],
                out_path=out_npz,
                n_workers=args.workers,
            )
            print(f'  Variante {suffix} completata in {_fmt(time.time() - t0)}')

    print(f'\n{"═" * 60}')
    print(f'  Tutto completato in {_fmt(time.time() - t_global)}')
    print(f'{"═" * 60}')


if __name__ == '__main__':
    main()
