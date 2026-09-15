#!/usr/bin/env python3
"""
Script autonomo per la generazione dei dataset CSI (H_activity) su tutte le
6 attività (IDLE, WALK, RUN, JUMP, FALL, STAND), simulata multi-RX e reale.

Versione v2: granularità 1 job per variante (non-IDLE), default jobs-per-gpu=3.
Ogni attività non-IDLE genera N_SAMPLES job indipendenti → GPU sature con molti worker.
IDLE resta 1 job per RX (PathSolver eseguito una sola volta, 50 sample AWGN).

Esegue i job in parallelo su GPU disponibili tramite sottoprocessi auto-referenziati:
  - senza --worker → orchestratore (nessun import Sionna)
  - con   --worker ACTIVITY RX_IDX [--variant-idx N] [--real] → worker singola variante

Dataset generati (--dataset-version v3):
  Simulati (--sim):
    dataset_v3/              griglia 9×4, 36 RX, con RIS        (scena: ttilab_furnished_sim)
    dataset_v3_noris/        griglia 9×4, 36 RX, senza RIS

  Reali (--real):
    dataset_real_v3/         rx8 [4,-1.5,1.0], con RIS           (scena: ttilab_furnished_real)
    dataset_real_v3_noris/   rx8, senza RIS
    dataset_real_v3_out_of_grid/    rx NLOS [4,-1.0,1.0], con RIS
    dataset_real_noris/             rx NLOS, senza RIS
    dataset_real_v4/         centro [0.5,0.0,1.0], con RIS
    dataset_real_v4_noris/   centro, senza RIS
    dataset_real_v3_envv3/   rx8, con RIS                         (scena: ttilab_furnished_v3)
    dataset_real_v3_oog_envv3/      rx NLOS, con RIS
    dataset_real_v4_envv3/   centro, con RIS

  unknown_room ("alien room", 15m x 3m x 4m, geometria diversa da TTI-Lab,
  per testare la generalizzazione a un ambiente mai visto -- non è una
  griglia di training, è una coppia baseline/target a punto fisso NLOS,
  vedi TX/RIS/RX_POS_UNKNOWN in cima al file):
    dataset_unknown_room_baseline/  rx NLOS fisso, con RIS   (scena: unknown_room_baseline_activity)
    dataset_unknown_room_target/    rx NLOS fisso, con RIS   (scena: unknown_room_target_activity, +oggetti metallici)

Esempi:
    # simulato (train) — ambiente semplificato, griglia 36 RX
    python run_acquisition_parallel_v2.py --sim  --dataset-version v3

    # reale classico — ambiente ttilab_furnished_real, rx8
    python run_acquisition_parallel_v2.py --real --dataset-version v3

    # reale senza RIS
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --no-ris

    # reale posizione centro stanza
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --real-pos v4

    # reale con ambiente v3 (ttilab_furnished_v3, monitor/PC/finestre/porte), rx8
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --scene-env real_v3

    # reale con ambiente v3, posizione centro
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --scene-env real_v3 --real-pos v4

    # solo alcune attività
    python run_acquisition_parallel_v2.py --sim --dataset-version v3 --activities WALK,RUN

    # GPU specifiche, più parallelismo
    python run_acquisition_parallel_v2.py --sim --dataset-version v3 --gpus 1,2,3 --jobs-per-gpu 2

    # dry-run (stampa piano senza eseguire)
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --scene-env real_v3 --dry-run

    # unknown_room: baseline (sim, punto fisso NLOS, rx_idx ignorato -> sempre rx0)
    python run_acquisition_parallel_v2.py --sim  --dataset-version v3 --scene-env unknown_sim --rx 0

    # unknown_room: target (con oggetti metallici aggiuntivi)
    python run_acquisition_parallel_v2.py --real --dataset-version v3 --scene-env unknown_real
"""

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# ─── Costanti condivise orchestratore + worker ────────────────────────────────
FREQ    = 1970e6          # Hz
TX_POS  = [-1.0, 1.89, 1.8]

RX_POSITIONS_SIM = [
    [4,  -1.0, 1.0],
    [4,   1.0, 1.0],
    [-4, -1.0, 1.0],
    [0,    0,  1.0],
]
RX_POS_REAL = [4, -1.0, 1.0]

RIS_POS        = [4.4, 1.0, 1.5]
RIS_ROWS       = 1
RIS_COLS       = 1
RIS_PANEL_ROWS = 4
RIS_PANEL_COLS = 4
N_ANT          = RIS_ROWS * RIS_COLS
N_PANELS       = RIS_PANEL_ROWS * RIS_PANEL_COLS

PERSON_H = 1.75
PERSON_Z = PERSON_H / 2

MAX_DEPTH_GNB    = 10
MAX_DEPTH_RELAY  = 5
PANEL_GROUP_SIZE = 4
SNR_DB           = 18.0
N_STEPS          = 40
N_SAMPLES        = 50
N_SC             = 1272

X_ZONES   = [(1.5, 2.7), (0.0, 1.3), (-3.0, -2.0)]
RNG_SEEDS = {'IDLE': 42, 'WALK': 123, 'RUN': 789, 'JUMP': 456, 'FALL': 123, 'STAND': 456}

# ── unknown_room: 15m x 3m x 4m "alien" room, deliberately different from
# TTI-Lab, used to test generalization to an unseen room geometry (see
# indoor_envs/unknown_room_baseline/generate_shell.py for the full layout).
# gNB near the north wall of the large room; RIS on the east wall of the
# small room; RX target in NLOS, shielded by the partition (x=[2.5,2.7]).
TX_POS_UNKNOWN         = [-1.5, 1.2, 1.8]
RIS_POS_UNKNOWN        = [7.3, 0.8, 1.5]
RX_POS_UNKNOWN_NLOS    = [6.0, -1.0, 1.0]
RX_POS_UNKNOWN_LOS     = [1.0, -1.0, 1.0]   # reference LOS point, large room

# Activity zones for unknown_room: the room is only 3m deep, and a FALL's
# centre can move up to PERSON_H/2 (~0.875m) from its nominal (x,y) in *any*
# direction (see make_fall_trajectory below), so FALL needs a much larger
# clearance from walls/furniture than WALK/RUN/JUMP/STAND. Furniture is
# clustered near the short ends of the room (indoor_envs/unknown_room_baseline
# /generate_furniture.py), leaving these X ranges as clear, full-depth gaps:
X_ZONES_UNKNOWN        = [(-5.4, -1.0), (-1.0, 2.2), (3.0, 6.3)]
# Y range for WALK/RUN/JUMP/STAND: 0.3m clear of both walls (y=[-1.5,1.5])
ACTIVITY_Y_RANGE_UNKNOWN     = (-1.2, 1.2)
WALK_Y_START_RANGE_UNKNOWN   = (-1.0, -0.3)
WALK_Y_END_RANGE_UNKNOWN     = (0.3, 1.0)
# FALL needs its own, smaller X zones (>=0.875m + margin from furniture/
# partition) and a tight Y range centred in the room:
FALL_X_ZONES_UNKNOWN   = [(-4.75, 1.55), (3.65, 5.65)]
FALL_Y_RANGE_UNKNOWN   = (-0.4, 0.4)

SCENE_PATHS = {
    'sim': {
        'default': 'indoor_envs/ttilab_furnished_sim/ttilab_furnished_sim.xml',
        'IDLE':    'indoor_envs/ttilab_furnished/ttilab_furnished.xml',
    },
    'real': {
        'default': 'indoor_envs/ttilab_furnished_real/ttilab_furnished_real.xml',
        'IDLE':    'indoor_envs/ttilab_furnished_v2/ttilab_furnished_v2.xml',
    },
    'real_v3': {
        'default': 'indoor_envs/ttilab_furnished_v3/ttilab_furnished_v3_activity.xml',
        'IDLE':    'indoor_envs/ttilab_furnished_v3/ttilab_furnished_v3.xml',
    },
    'unknown_sim': {
        'default': 'indoor_envs/unknown_room_baseline_activity/unknown_room_baseline_activity.xml',
        'IDLE':    'indoor_envs/unknown_room_baseline/unknown_room_baseline.xml',
    },
    'unknown_real': {
        'default': 'indoor_envs/unknown_room_target_activity/unknown_room_target_activity.xml',
        'IDLE':    'indoor_envs/unknown_room_target/unknown_room_target.xml',
    },
}

DIR_DATASET_REAL="dataset_real_v2"
DIR_DATASET="dataset_v2"

SIM_DIRS = {
    'IDLE':  f'{DIR_DATASET}/furnished_idle',
    'WALK':  f'{DIR_DATASET}/walk_furnished',
    'RUN':   f'{DIR_DATASET}/run_furnished',
    'JUMP':  f'{DIR_DATASET}/jump_furnished',
    'FALL':  f'{DIR_DATASET}/fall_furnished',
    'STAND': f'{DIR_DATASET}/stand_furnished',
}
REAL_DIRS = {
    'IDLE':  f'{DIR_DATASET_REAL}/furnished_idle',
    'WALK':  f'{DIR_DATASET_REAL}/walk_furnished',
    'RUN':   f'{DIR_DATASET_REAL}/run_furnished',
    'JUMP':  f'{DIR_DATASET_REAL}/jump_furnished',
    'FALL':  f'{DIR_DATASET_REAL}/fall_furnished',
    'STAND': f'{DIR_DATASET_REAL}/stand_furnished',
}

DIR_DATASET_V3      = "dataset_v3"
DIR_DATASET_REAL_V3 = "dataset_real_v3"

# Griglia 9×4, passo 1m, z=1.0m — 36 posizioni: rx0..rx35
# X: -4,-3,...,4  (9 valori centrati)
# Y: -1.5,-0.5,0.5,1.5  (4 valori centrati, 0.5m dai muri ±2.0m)
RX_POSITIONS_SIM_V3 = [
    [float(x), y, 1.0]
    for y in [-1.5, -0.5, 0.5, 1.5]
    for x in range(-4, 5)
]  # 36 posizioni

SIM_DIRS_V3 = {
    act: d.replace(DIR_DATASET, DIR_DATASET_V3)
    for act, d in SIM_DIRS.items()
}
REAL_DIRS_V3 = {
    act: d.replace(DIR_DATASET_REAL, DIR_DATASET_REAL_V3)
    for act, d in REAL_DIRS.items()
}

RX_POS_REAL_V3 = RX_POSITIONS_SIM_V3[8]   # [4, -1.5, 1.0] — rx8, allineato alla griglia v3

DIR_DATASET_REAL_V3_OOG = "dataset_real_v3_out_of_grid"
REAL_DIRS_V3_OOG = {
    act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V3_OOG)
    for act, d in REAL_DIRS_V3.items()
}

RX_POS_REAL_V4 = [0.5, 0.0, 1.0]   # centro del quadrato rx13,rx14,rx22,rx23

DIR_DATASET_REAL_V4 = "dataset_real_v4"
REAL_DIRS_V4 = {
    act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V4)
    for act, d in REAL_DIRS_V3.items()
}

# ── No-RIS: stessa griglia v3, senza relay RIS ───────────────────────────────
DIR_DATASET_V3_NORIS      = "dataset_v3_noris"
SIM_DIRS_V3_NORIS         = {act: d.replace(DIR_DATASET_V3, DIR_DATASET_V3_NORIS)
                              for act, d in SIM_DIRS_V3.items()}

DIR_DATASET_REAL_V3_NORIS = "dataset_real_v3_noris"   # rx8 grid + no RIS
REAL_DIRS_V3_NORIS        = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V3_NORIS)
                              for act, d in REAL_DIRS_V3.items()}

DIR_DATASET_REAL_NORIS    = "dataset_real_noris"       # out-of-grid (NLOS) + no RIS
REAL_DIRS_NORIS_OOG       = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_NORIS)
                              for act, d in REAL_DIRS_V3.items()}

DIR_DATASET_REAL_V4_NORIS = "dataset_real_v4_noris"   # centro quadrato + no RIS
REAL_DIRS_V4_NORIS        = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V4_NORIS)
                              for act, d in REAL_DIRS_V3.items()}

# ── Ambiente v3 (ttilab_furnished_v3): directory separate per scena più ricca ─
DIR_DATASET_REAL_V3_ENVV3     = "dataset_real_v3_envv3"       # rx8 grid
REAL_DIRS_V3_ENVV3            = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V3_ENVV3)
                                  for act, d in REAL_DIRS_V3.items()}
DIR_DATASET_REAL_V3_OOG_ENVV3 = "dataset_real_v3_oog_envv3"   # out-of-grid
REAL_DIRS_V3_OOG_ENVV3        = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V3_OOG_ENVV3)
                                  for act, d in REAL_DIRS_V3.items()}
DIR_DATASET_REAL_V4_ENVV3     = "dataset_real_v4_envv3"        # centro quadrato
REAL_DIRS_V4_ENVV3            = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_REAL_V4_ENVV3)
                                  for act, d in REAL_DIRS_V3.items()}

# ── unknown_room: baseline/target dataset directories (fixed-point eval,
# not a training grid -- see TX/RIS/RX_POS_UNKNOWN above) ────────────────────
DIR_DATASET_UNKNOWN_SIM  = "dataset_unknown_room_baseline"
DIR_DATASET_UNKNOWN_REAL = "dataset_unknown_room_target"
SIM_DIRS_UNKNOWN  = {act: d.replace(DIR_DATASET_V3, DIR_DATASET_UNKNOWN_SIM)
                      for act, d in SIM_DIRS_V3.items()}
REAL_DIRS_UNKNOWN = {act: d.replace(DIR_DATASET_REAL_V3, DIR_DATASET_UNKNOWN_REAL)
                      for act, d in REAL_DIRS_V3.items()}

ACTIVITY_ORDER     = ['IDLE', 'WALK', 'RUN', 'JUMP', 'FALL', 'STAND']
PATH_CSI_D1        = 'csi_matlab/csirs_data_density1.mat'
DEFAULT_PYTHON_BIN = sys.executable
BASE_DIR           = Path(__file__).resolve().parent

# ─── Materiali (necessari nel worker) ─────────────────────────────────────────
_MAT_COLORS = {
    'concrete': [0.5,  0.5,  0.5],
    'metal':    [0.75, 0.75, 0.75],
    'wood':     [0.55, 0.27, 0.07],
    'human':    [0.8,  0.6,  0.4],
    'glass':    [0.6,  0.8,  0.9],
}


def _get_material_props(freq_hz):
    f = freq_hz / 1e9
    return {
        'itu-concrete': {'eps': 1.0,  'sig': 0},
        'itu-brick':    {'eps': 3.75, 'sig': 0.038},
        'itu-glass':    {'eps': 6.27, 'sig': 0.0043 * f ** 1.1925},
        'glass':        {'eps': 6.27, 'sig': 0.0043 * f ** 1.1925},
        'itu-wood':     {'eps': 1.99, 'sig': 0.0047 * f ** 1.0718},
        'itu-metal':    {'eps': 1.0,  'sig': 1e7},
        'itu-human':    {'eps': 60.0, 'sig': 1.5},
        'metal':        {'eps': 1.0,  'sig': 1e7},
        'concrete':     {'eps': 5.31, 'sig': 0.0326 * f ** 0.8095},
        'wood':         {'eps': 1.99, 'sig': 0.0047 * f ** 1.0718},
        'marble':       {'eps': 7.0,  'sig': 0.02},
        'human':        {'eps': 60.0, 'sig': 1.5},
    }


def _apply_material_fix(scene, freq_hz, RadioMaterial):
    fix_props = _get_material_props(freq_hz)
    old_names = list(scene.radio_materials.keys())
    m2m = {}
    for name in old_names:
        old  = scene.get(name)
        base = (name
                .replace('mat-itu_', '').replace('mat-itu-', '')
                .replace('itu_', '').replace('itu-', '')
                .replace('mat-', ''))
        lookup = name if name in fix_props else base
        if lookup in fix_props:
            eps, sig = fix_props[lookup]['eps'], fix_props[lookup]['sig']
        else:
            try:
                eps = float(old.relative_permittivity.numpy())
                sig = float(old.conductivity.numpy())
            except Exception:
                eps, sig = 5.0, 0.01
        nm = RadioMaterial(f'{name}_fix', relative_permittivity=eps, conductivity=sig)
        if base in _MAT_COLORS:
            nm.color = _MAT_COLORS[base]
        scene.add(nm)
        m2m[name] = nm
    for obj in scene.objects.values():
        if obj.radio_material.name in m2m:
            obj.radio_material = m2m[obj.radio_material.name]
    for name in old_names:
        scene.remove(name)


# ─── Traiettorie ──────────────────────────────────────────────────────────────
# Tutte restituiscono (xs, ys, zs, orients, vels):
#   xs, ys, zs  : np.array float (N_STEPS,)
#   orients     : list of (alpha, beta, gamma) o None (se orientazione fissa/verticale)
#   vels        : list of [vx, vy, vz]

def _sample_x(rng, zones=X_ZONES):
    lo, hi = zones[int(rng.integers(0, len(zones)))]
    return float(rng.uniform(lo, hi))


def make_walk_trajectory(person_x, y_start, y_end, n_rounds=2, steps_per_half=10):
    import numpy as np
    one_y = np.concatenate([
        np.linspace(y_start, y_end,   steps_per_half),
        np.linspace(y_end,   y_start, steps_per_half),
    ])
    one_v = [[0., 1.2, 0.]] * steps_per_half + [[0., -1.2, 0.]] * steps_per_half
    ys   = np.tile(one_y, n_rounds)
    vels = one_v * n_rounds
    n    = len(ys)
    xs   = np.full(n, float(person_x))
    zs   = np.full(n, PERSON_Z)
    return xs, ys, zs, [None] * n, vels


def make_run_trajectory(person_x, y_start, y_end, n_rounds=4, steps_per_half=5, velocity=3.0):
    import numpy as np
    one_y = np.concatenate([
        np.linspace(y_start, y_end,   steps_per_half),
        np.linspace(y_end,   y_start, steps_per_half),
    ])
    one_v = [[0., velocity, 0.]] * steps_per_half + [[0., -velocity, 0.]] * steps_per_half
    ys   = np.tile(one_y, n_rounds)
    vels = one_v * n_rounds
    n    = len(ys)
    xs   = np.full(n, float(person_x))
    zs   = np.full(n, PERSON_Z)
    return xs, ys, zs, [None] * n, vels


def make_jump_trajectory(person_x, person_y,
                          n_jumps=5, jump_height=0.4):
    import numpy as np
    n_total    = N_STEPS                        # 40
    n_per_jump = n_total // n_jumps             # 8 step per salto
    t_one  = np.linspace(0, np.pi, n_per_jump)
    z_one  = PERSON_Z + jump_height * np.sin(t_one)
    zs     = np.tile(z_one, n_jumps)[:n_total]
    xs     = np.full(n_total, float(person_x))
    ys     = np.full(n_total, float(person_y))
    orients = [(0., 0., 0.)] * n_total
    vels    = [[0., 0., float(zs[min(i + 1, n_total - 1)] - zs[i])]
               for i in range(n_total)]
    return xs, ys, zs, orients, vels


_FALL_Z_END = 0.15


def make_fall_trajectory(person_x, person_y, fall_phi,
                          z_end=_FALL_Z_END, n_stand=1, n_fall=37, n_ground=2):
    import numpy as np
    n_total = n_stand + n_fall + n_ground
    thetas  = np.concatenate([
        np.zeros(n_stand),
        np.linspace(0, np.pi / 2, n_fall),
        np.full(n_ground, np.pi / 2),
    ])
    xs      = person_x + PERSON_H / 2 * np.sin(thetas) * np.cos(fall_phi)
    ys      = person_y + PERSON_H / 2 * np.sin(thetas) * np.sin(fall_phi)
    zs      = z_end + (PERSON_H / 2 - z_end) * np.cos(thetas)
    orients = [(fall_phi, float(t), 0.) for t in thetas]
    vels    = [[0., 0., 0.]] * n_total
    return xs, ys, zs, orients, vels


_STAND_RADIUS = 0.10


def make_stand_trajectory(center_x, center_y, radius=_STAND_RADIUS,
                           n_steps=N_STEPS, seed=None):
    import numpy as np
    rng   = np.random.default_rng(seed)
    r     = radius * np.sqrt(rng.uniform(0, 1, n_steps))
    theta = rng.uniform(0, 2 * np.pi, n_steps)
    xs    = center_x + r * np.cos(theta)
    ys    = center_y + r * np.sin(theta)
    zs    = np.full(n_steps, PERSON_Z)
    vels  = [[0., 0., 0.]] * n_steps
    return xs, ys, zs, [None] * n_steps, vels


def build_variants(activity, rng, mode=None):
    # unknown_room is much narrower (3m) than TTI-Lab (4m) and uses its own,
    # more conservative activity zones -- see X_ZONES_UNKNOWN and
    # FALL_X_ZONES_UNKNOWN/FALL_Y_RANGE_UNKNOWN above for the geometric
    # rationale (FALL needs extra clearance because the body's centre can
    # move up to PERSON_H/2 from its nominal position while falling).
    is_unknown = mode in ('unknown_sim', 'unknown_real')
    zones      = X_ZONES_UNKNOWN if is_unknown else X_ZONES
    y_lo, y_hi = ACTIVITY_Y_RANGE_UNKNOWN if is_unknown else (-2.0, 2.0)
    walk_y_start = WALK_Y_START_RANGE_UNKNOWN if is_unknown else (-2.0, -0.5)
    walk_y_end   = WALK_Y_END_RANGE_UNKNOWN   if is_unknown else (0.5, 2.0)
    fall_zones   = FALL_X_ZONES_UNKNOWN if is_unknown else X_ZONES
    fall_y_lo, fall_y_hi = FALL_Y_RANGE_UNKNOWN if is_unknown else (-2.0, 2.0)

    if activity == 'WALK':
        return [
            {
                'x':       _sample_x(rng, zones),
                'y_start': float(rng.uniform(*walk_y_start)),
                'y_end':   float(rng.uniform(*walk_y_end)),
            }
            for _ in range(N_SAMPLES)
        ]
    if activity == 'RUN':
        return [
            {
                'x':       _sample_x(rng, zones),
                'y_start': float(rng.uniform(*walk_y_start)),
                'y_end':   float(rng.uniform(*walk_y_end)),
            }
            for _ in range(N_SAMPLES)
        ]
    if activity == 'JUMP':
        return [
            {
                'x': _sample_x(rng, zones),
                'y': float(rng.uniform(y_lo, y_hi)),
            }
            for _ in range(N_SAMPLES)
        ]
    if activity == 'FALL':
        import numpy as np
        return [
            {
                'x':   _sample_x(rng, fall_zones),
                'y':   float(rng.uniform(fall_y_lo, fall_y_hi)),
                'phi': float(rng.uniform(0, 2 * np.pi)),
            }
            for _ in range(N_SAMPLES)
        ]
    if activity == 'STAND':
        return [
            {
                'cx':   _sample_x(rng, zones),
                'cy':   float(rng.uniform(y_lo, y_hi)),
                'seed': int(rng.integers(0, 10000)),
            }
            for _ in range(N_SAMPLES)
        ]
    raise ValueError(f'Attività sconosciuta: {activity}')


def make_trajectory(activity, variant):
    if activity == 'WALK':
        return make_walk_trajectory(variant['x'], variant['y_start'], variant['y_end'])
    if activity == 'RUN':
        return make_run_trajectory(variant['x'], variant['y_start'], variant['y_end'])
    if activity == 'JUMP':
        return make_jump_trajectory(variant['x'], variant['y'])
    if activity == 'FALL':
        return make_fall_trajectory(variant['x'], variant['y'], variant['phi'])
    if activity == 'STAND':
        return make_stand_trajectory(variant['cx'], variant['cy'], seed=variant['seed'])
    raise ValueError(f'Attività sconosciuta: {activity}')


# ─── Worker — logica Sionna (tutti gli import pesanti qui dentro) ─────────────

def run_worker(activity, rx_idx, real, variant_idx=None, dataset_version='v2', real_pos='grid', use_ris=True, scene_env=None):
    """Eseguito nel sottoprocesso. Importa Sionna e genera i CSV per un job."""
    import numpy as np
    import scipy.io
    import torch
    from scipy.interpolate import interp1d
    import sionna.rt
    from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                            PathSolver, RadioMaterial, subcarrier_frequencies)
    from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel
    from sionna.phy.ofdm import ResourceGrid, PilotPattern, ResourceGridMapper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{activity} rx{rx_idx}{"r" if real else "s"}] Device: {device}')

    # ── Percorsi ───────────────────────────────────────────────────────────
    mode     = scene_env if scene_env else ('real' if real else 'sim')
    is_unknown = mode in ('unknown_sim', 'unknown_real')
    tx_pos     = TX_POS_UNKNOWN  if is_unknown else TX_POS
    ris_pos    = RIS_POS_UNKNOWN if is_unknown else RIS_POS
    if mode in ('unknown_sim', 'unknown_real'):
        # unknown_room: fixed-point baseline/target pair (not a training
        # grid), single NLOS receiver -- see TX/RIS/RX_POS_UNKNOWN above.
        dirs   = SIM_DIRS_UNKNOWN if mode == 'unknown_sim' else REAL_DIRS_UNKNOWN
        rx_pos = RX_POS_UNKNOWN_NLOS
    elif dataset_version == 'v3':
        if real:
            if scene_env == 'real_v3':
                if real_pos == 'out_of_grid':
                    dirs, rx_pos = REAL_DIRS_V3_OOG_ENVV3, RX_POS_REAL
                elif real_pos == 'v4':
                    dirs, rx_pos = REAL_DIRS_V4_ENVV3, RX_POS_REAL_V4
                else:
                    dirs, rx_pos = REAL_DIRS_V3_ENVV3, RX_POS_REAL_V3
            elif real_pos == 'out_of_grid':
                dirs, rx_pos = (REAL_DIRS_NORIS_OOG if not use_ris else REAL_DIRS_V3_OOG), RX_POS_REAL
            elif real_pos == 'v4':
                dirs, rx_pos = (REAL_DIRS_V4_NORIS  if not use_ris else REAL_DIRS_V4), RX_POS_REAL_V4
            else:
                dirs, rx_pos = (REAL_DIRS_V3_NORIS  if not use_ris else REAL_DIRS_V3), RX_POS_REAL_V3
        else:
            dirs   = SIM_DIRS_V3_NORIS if not use_ris else SIM_DIRS_V3
            rx_pos = RX_POSITIONS_SIM_V3[rx_idx]
    else:
        dirs   = REAL_DIRS if real else SIM_DIRS
        rx_pos = RX_POS_REAL if real else RX_POSITIONS_SIM[rx_idx]
    scene_key = SCENE_PATHS[mode]
    scene_path = str(BASE_DIR / scene_key.get(activity, scene_key['default']))

    if real:
        out_dir  = str(BASE_DIR / dirs[activity])
        idle_dir = str(BASE_DIR / dirs['IDLE'])
    else:
        out_dir  = str(BASE_DIR / dirs[activity] / f'rx{rx_idx}')
        idle_dir = str(BASE_DIR / dirs['IDLE']    / f'rx{rx_idx}')
    os.makedirs(out_dir, exist_ok=True)

    # ── Panel positions ────────────────────────────────────────────────────
    _lambda  = 3e8 / FREQ
    _d_elem  = 0.25 * _lambda
    _pitch_y = RIS_COLS * _d_elem
    _pitch_z = RIS_ROWS * _d_elem
    panel_positions = []
    for _row in range(RIS_PANEL_ROWS):
        for _col in range(RIS_PANEL_COLS):
            _y = ris_pos[1] + (_col - (RIS_PANEL_COLS - 1) / 2) * _pitch_y
            _z = ris_pos[2] + (_row - (RIS_PANEL_ROWS - 1) / 2) * _pitch_z
            panel_positions.append([ris_pos[0], float(_y), float(_z)])

    # ── ResourceGrid ───────────────────────────────────────────────────────
    tx_D1        = scipy.io.loadmat(str(BASE_DIR / PATH_CSI_D1))['txGrid'].T.astype(np.complex64)
    mask_D1_np   = (np.abs(tx_D1) > 1e-10)
    csirs_mask   = mask_D1_np.copy()
    PILOT_SYMBOL_INDICES = np.where(csirs_mask.any(axis=1))[0].tolist()

    mask_t   = torch.tensor(csirs_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    pilots_t = torch.tensor(tx_D1[csirs_mask], dtype=torch.complex64).unsqueeze(0).unsqueeze(0)
    pilot_pattern = PilotPattern(mask=mask_t, pilots=pilots_t)

    rg = ResourceGrid(
        num_ofdm_symbols=14, fft_size=N_SC, subcarrier_spacing=15e3,
        num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=90,
        pilot_pattern=pilot_pattern,
    )
    rg_mapper  = ResourceGridMapper(rg)
    x_payload  = torch.zeros(1, 1, 1, rg.num_data_symbols, dtype=torch.complex64).to(device)
    x_rg_final = rg_mapper(x_payload)

    snr_linear    = 10 ** (SNR_DB / 10)
    no            = 1.0 / snr_linear
    frequencies   = torch.tensor(subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing),
                                  dtype=torch.float32)
    channel_apply = ApplyOFDMChannel(add_awgn=True)
    sc_all        = np.arange(N_SC)

    # ── Helper: estrai H (N_SC,) da un paths object ────────────────────────
    def _compute_H_step_from_paths(paths):
        a, tau = paths.cir(normalize_delays=False, out_type='numpy')
        if a.ndim == 5 and a.shape[3] > 1:
            a = a.sum(axis=3, keepdims=True) / np.sqrt(a.shape[3])
            if tau.ndim == 4 and tau.shape[2] > 1:
                tau = tau[:, :, :1, :]
        a_t   = torch.tensor(a,   dtype=torch.complex64).unsqueeze(0).to(device)
        tau_t = torch.tensor(tau, dtype=torch.float32).unsqueeze(0).to(device)
        h_f   = cir_to_ofdm_channel(frequencies, a_t, tau_t, normalize=True)
        y_w   = channel_apply(x_rg_final, h_f, no)
        y_np  = y_w[0, 0, 0].cpu().numpy()
        tx_np = x_rg_final[0, 0, 0].cpu().numpy()
        H_parts = []
        for sym in PILOT_SYMBOL_INDICES:
            m    = csirs_mask[sym]
            H_ls = y_np[sym, m] / tx_np[sym, m]
            sc_p = np.where(m)[0]
            f_amp = interp1d(sc_p, np.abs(H_ls),             kind='linear', fill_value='extrapolate')
            f_pha = interp1d(sc_p, np.unwrap(np.angle(H_ls)), kind='linear', fill_value='extrapolate')
            H_parts.append(f_amp(sc_all) * np.exp(1j * f_pha(sc_all)))
        return np.mean(H_parts, axis=0).astype(np.complex64)

    # ── Helper: H (N_SC,) da h_f già calcolato (per AWGN-only loop) ────────
    def _compute_H_step_from_hf(h_f):
        y_w   = channel_apply(x_rg_final, h_f, no)
        y_np  = y_w[0, 0, 0].cpu().numpy()
        tx_np = x_rg_final[0, 0, 0].cpu().numpy()
        H_parts = []
        for sym in PILOT_SYMBOL_INDICES:
            m    = csirs_mask[sym]
            H_ls = y_np[sym, m] / tx_np[sym, m]
            sc_p = np.where(m)[0]
            f_amp = interp1d(sc_p, np.abs(H_ls),             kind='linear', fill_value='extrapolate')
            f_pha = interp1d(sc_p, np.unwrap(np.angle(H_ls)), kind='linear', fill_value='extrapolate')
            H_parts.append(f_amp(sc_all) * np.exp(1j * f_pha(sc_all)))
        return np.mean(H_parts, axis=0).astype(np.complex64)

    # ── Helper: CSV row ────────────────────────────────────────────────────
    def H_to_row(H_cx):
        return ';'.join(
            f"{H_cx[i].real:.6f},{H_cx[i].imag:.6f},"
            f"{np.abs(H_cx[i]):.6f},{np.angle(H_cx[i]):.6f}"
            for i in range(len(H_cx))
        )

    sc_header = ';'.join(str(i) for i in range(N_SC))

    # ── GNB variant per-step ───────────────────────────────────────────────
    def run_gnb_variant(scene, p_solver, xs, ys, zs, orients, vels, label=''):
        person  = scene.get('person')
        n_steps = len(ys)
        H_list  = []
        for i, (px, py, pz, orient, vel) in enumerate(zip(xs, ys, zs, orients, vels)):
            if orient is not None:
                person.orientation = orient
            person.position = [float(px), float(py), float(pz)]
            person.velocity = vel
            paths = p_solver(
                scene=scene, max_depth=MAX_DEPTH_GNB,
                los=True, specular_reflection=True,
                diffuse_reflection=True, refraction=True,
                synthetic_array=False, seed=41,
            )
            H_list.append(_compute_H_step_from_paths(paths))
            print(f'  [{label}] gNB step {i+1:3d}/{n_steps}', end='\r')
        print()
        return np.array(H_list)

    # ── Relay variant per-step ─────────────────────────────────────────────
    def run_relay_variant(scene, p_solver, xs, ys, zs, orients, vels, label=''):
        person  = scene.get('person')
        n_steps = len(ys)
        H_list  = []
        for i, (px, py, pz, orient, vel) in enumerate(zip(xs, ys, zs, orients, vels)):
            if orient is not None:
                person.orientation = orient
            person.position = [float(px), float(py), float(pz)]
            person.velocity = vel
            H_step = np.zeros(N_SC, dtype=np.complex64)
            for group_start in range(0, len(panel_positions), PANEL_GROUP_SIZE):
                group = list(range(group_start,
                                   min(group_start + PANEL_GROUP_SIZE, len(panel_positions))))
                for idx in group:
                    scene.add(Transmitter(name=f'TX_RIS_{idx:02d}',
                                          position=panel_positions[idx]))
                paths = p_solver(
                    scene=scene, max_depth=MAX_DEPTH_RELAY,
                    los=True, specular_reflection=True,
                    diffuse_reflection=True, refraction=True,
                    synthetic_array=True, seed=41,
                )
                a, tau = paths.cir(normalize_delays=False, out_type='numpy')
                for local_i in range(len(group)):
                    a_i   = a[:, :, local_i:local_i + 1, :, :]
                    if a_i.shape[3] > 1:
                        a_i = a_i.sum(axis=3, keepdims=True) / np.sqrt(N_ANT)
                    tau_i = (tau[:, local_i:local_i + 1, :1, :] if tau.ndim == 4
                             else tau[:, local_i:local_i + 1, :])
                    a_t   = torch.tensor(a_i,   dtype=torch.complex64).unsqueeze(0).to(device)
                    tau_t = torch.tensor(tau_i, dtype=torch.float32).unsqueeze(0).to(device)
                    h_f   = cir_to_ofdm_channel(frequencies, a_t, tau_t, normalize=True)
                    H_step += _compute_H_step_from_hf(h_f)
                for idx in group:
                    scene.remove(f'TX_RIS_{idx:02d}')
                torch.cuda.empty_cache()
            H_list.append(H_step)
            print(f'  [{label}] RIS step {i+1:3d}/{n_steps}', end='\r')
        print()
        return np.array(H_list)

    # ── load_H_prior: legge v0 e fa media sugli step ──────────────────────
    def load_H_prior(idle_dir_path):
        with open(os.path.join(idle_dir_path, 'H_activity_v0.csv')) as f:
            f.readline()
            H_steps = []
            for line in f:
                parts = line.strip().split(';')
                H_steps.append(np.array([
                    complex(float(v.split(',')[0]), float(v.split(',')[1]))
                    for v in parts[1:]
                ], dtype=np.complex64))
        return np.mean(H_steps, axis=0)

    # ── Carica scena helper ────────────────────────────────────────────────
    def load_and_setup_scene(scene_path_str):
        scene = load_scene(scene_path_str, merge_shapes=False)
        _apply_material_fix(scene, FREQ, RadioMaterial)
        scene.frequency = FREQ
        scene.rx_array  = PlanarArray(num_rows=1, num_cols=1, pattern='dipole', polarization='V')
        scene.add(Receiver(name='RX', position=rx_pos))
        return scene

    # ══════════════════════════════════════════════════════════════════════
    # IDLE: paths calcolati UNA VOLTA, loop N_SAMPLES × N_STEPS solo AWGN
    # ══════════════════════════════════════════════════════════════════════
    if activity == 'IDLE':
        print('Caricamento scena IDLE...')
        scene_ris = load_and_setup_scene(scene_path)
        p_solver  = PathSolver()

        # ── PathSolver gNB — una volta sola ───────────────────────────────
        print('PathSolver gNB...')
        scene_ris.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                          pattern='dipole', polarization='V')
        scene_ris.add(Transmitter(name='TX', position=tx_pos))
        paths_gnb = p_solver(
            scene=scene_ris, max_depth=MAX_DEPTH_GNB,
            los=True, specular_reflection=True,
            diffuse_reflection=True, refraction=True,
            synthetic_array=False, seed=41,
        )
        a_gnb, tau_gnb = paths_gnb.cir(normalize_delays=False, out_type='numpy')
        if a_gnb.ndim == 5 and a_gnb.shape[3] > 1:
            a_gnb = a_gnb.sum(axis=3, keepdims=True) / np.sqrt(a_gnb.shape[3])
            if tau_gnb.ndim == 4 and tau_gnb.shape[2] > 1:
                tau_gnb = tau_gnb[:, :, :1, :]
        a_t   = torch.tensor(a_gnb,   dtype=torch.complex64).unsqueeze(0).to(device)
        tau_t = torch.tensor(tau_gnb, dtype=torch.float32).unsqueeze(0).to(device)
        hf_gnb = cir_to_ofdm_channel(frequencies, a_t, tau_t, normalize=True)
        scene_ris.remove('TX')
        torch.cuda.empty_cache()
        print('  hf_gnb calcolato')

        # ── PathSolver RIS — a gruppi per evitare OOM ─────────────────────
        hf_panels = []
        if use_ris:
            print('PathSolver RIS panels (a gruppi)...')
            scene_ris.tx_array = PlanarArray(num_rows=RIS_ROWS, num_cols=RIS_COLS,
                                              pattern='dipole', polarization='V')
            for group_start in range(0, N_PANELS, PANEL_GROUP_SIZE):
                group_end = min(group_start + PANEL_GROUP_SIZE, N_PANELS)
                group     = list(range(group_start, group_end))
                print(f'  Gruppo panel {group_start}..{group_end - 1}', end='\r')
                for idx in group:
                    scene_ris.add(Transmitter(name=f'TX_RIS_{idx:02d}',
                                               position=panel_positions[idx]))
                paths_ris = p_solver(
                    scene=scene_ris, max_depth=MAX_DEPTH_RELAY,
                    los=True, specular_reflection=True,
                    diffuse_reflection=True, refraction=True,
                    synthetic_array=True, seed=41,
                )
                a_ris, tau_ris = paths_ris.cir(normalize_delays=False, out_type='numpy')
                for local_i in range(len(group)):
                    a_i   = a_ris[:, :, local_i:local_i + 1, :, :]
                    if a_i.shape[3] > 1:
                        a_i = a_i.sum(axis=3, keepdims=True) / np.sqrt(N_ANT)
                    tau_i = (tau_ris[:, local_i:local_i + 1, :1, :] if tau_ris.ndim == 4
                             else tau_ris[:, local_i:local_i + 1, :])
                    a_t   = torch.tensor(a_i,   dtype=torch.complex64).unsqueeze(0).to(device)
                    tau_t = torch.tensor(tau_i, dtype=torch.float32).unsqueeze(0).to(device)
                    hf_panels.append(cir_to_ofdm_channel(frequencies, a_t, tau_t, normalize=True))
                for idx in group:
                    scene_ris.remove(f'TX_RIS_{idx:02d}')
                torch.cuda.empty_cache()
            print(f'\n  {len(hf_panels)} panel h_f calcolati')

        # ── Loop N_SAMPLES — solo AWGN varia ──────────────────────────────
        print(f'\nGenerazione {N_SAMPLES} sample × {N_STEPS} step...')
        for sample_id in range(N_SAMPLES):
            H_sample = np.zeros((N_STEPS, N_SC), dtype=np.complex64)
            for step in range(N_STEPS):
                H_row = _compute_H_step_from_hf(hf_gnb).copy()
                for hf in hf_panels:
                    H_row += _compute_H_step_from_hf(hf)
                H_sample[step] = H_row

            out_path = os.path.join(out_dir, f'H_activity_v{sample_id}.csv')
            with open(out_path, 'w') as fout:
                fout.write(f'step;{sc_header}\n')
                for step in range(N_STEPS):
                    fout.write(f'{step};{H_to_row(H_sample[step])}\n')
            print(f'  sample {sample_id + 1:3d}/{N_SAMPLES} salvato → {out_path}', end='\r')

        print(f'\nIDLE: {N_SAMPLES} sample salvati in {out_dir}/')
        return

    # ══════════════════════════════════════════════════════════════════════
    # Attività con traiettoria: WALK, RUN, JUMP, FALL, STAND
    # ══════════════════════════════════════════════════════════════════════
    rng      = np.random.default_rng(RNG_SEEDS[activity])
    variants = build_variants(activity, rng, mode=mode)

    H_prior = load_H_prior(idle_dir)
    print(f'H_prior caricato da {idle_dir}/H_activity_v0.csv')

    iter_indices = [variant_idx] if variant_idx is not None else range(N_SAMPLES)
    for v_idx in iter_indices:
        variant = variants[v_idx]
        print(f'\n── {activity} Variant {v_idx + 1}/{N_SAMPLES}  {variant}')

        xs, ys, zs, orients, vels = make_trajectory(activity, variant)
        n_steps = len(ys)

        scene_v = load_and_setup_scene(scene_path)
        p_solver_v = PathSolver()

        scene_v.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                        pattern='dipole', polarization='V')
        scene_v.add(Transmitter(name='TX', position=tx_pos))
        H_base = run_gnb_variant(scene_v, p_solver_v, xs, ys, zs, orients, vels,
                                  label=f'gnb_{activity.lower()}_v{v_idx}')
        scene_v.remove('TX')
        torch.cuda.empty_cache()

        if use_ris:
            scene_v.tx_array = PlanarArray(num_rows=RIS_ROWS, num_cols=RIS_COLS,
                                            pattern='dipole', polarization='V')
            H_ris = run_relay_variant(scene_v, p_solver_v, xs, ys, zs, orients, vels,
                                       label=f'ris_{activity.lower()}_v{v_idx}')
            H_cx_combined = H_base + H_ris
        else:
            H_cx_combined = H_base
        del scene_v, p_solver_v
        torch.cuda.empty_cache()
        H_activity    = H_cx_combined - H_prior

        out_combined = os.path.join(out_dir, f'H_combined_v{v_idx}.csv')
        with open(out_combined, 'w') as fout:
            fout.write(f'step;{sc_header}\n')
            for step in range(n_steps):
                fout.write(f'{step};{H_to_row(H_cx_combined[step])}\n')

        out_activity = os.path.join(out_dir, f'H_activity_v{v_idx}.csv')
        with open(out_activity, 'w') as fout:
            fout.write(f'step;{sc_header}\n')
            for step in range(n_steps):
                fout.write(f'{step};{H_to_row(H_activity[step])}\n')

        print(f'  Salvato: {out_activity}')

    print(f'\n{activity}: {N_SAMPLES} variant salvati in {out_dir}/')


# ─── Orchestratore ────────────────────────────────────────────────────────────

@dataclass
class Job:
    name:        str
    activity:    str
    real:        bool
    rx_idx:      int
    variant_idx: int = None   # None = tutte (solo IDLE); intero = singola variante
    depends_on:  str = None


def build_jobs(activities, rx_indices, do_sim, do_real, regenerate_idle=True):
    jobs = []
    if do_sim:
        for rx in rx_indices:
            idle_name = f'IDLE_sim_rx{rx}' if regenerate_idle else None
            if regenerate_idle:
                jobs.append(Job(idle_name, 'IDLE', False, rx))
            for act in activities:
                if act == 'IDLE':
                    continue
                for v in range(N_SAMPLES):
                    jobs.append(Job(f'{act}_sim_rx{rx}_v{v:03d}', act, False, rx,
                                    variant_idx=v, depends_on=idle_name))
    if do_real:
        idle_name = 'IDLE_real' if regenerate_idle else None
        if regenerate_idle:
            jobs.append(Job(idle_name, 'IDLE', True, 0))
        for act in activities:
            if act == 'IDLE':
                continue
            for v in range(N_SAMPLES):
                jobs.append(Job(f'{act}_real_v{v:03d}', act, True, 0,
                                variant_idx=v, depends_on=idle_name))
    return jobs


def detect_free_gpus(threshold_mib=8000):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            text=True,
        )
    except Exception as e:
        print(f'[WARN] nvidia-smi non disponibile ({e}), uso GPU 0 di default')
        return [0]
    gpus = []
    for line in out.strip().splitlines():
        idx, free = line.split(',')
        if int(free.strip()) >= threshold_mib:
            gpus.append(int(idx.strip()))
    return gpus or [0]


def _expected_output_file(job, dataset_version, real_pos, use_ris=True, scene_env=None):
    """Ritorna il Path del CSV atteso per questo job (per il check di resume)."""
    if scene_env in ('unknown_sim', 'unknown_real'):
        # Must mirror the mode == 'unknown_sim'/'unknown_real' branch in
        # run_worker() above, or resume/skip-check will look in the wrong dir.
        dirs = SIM_DIRS_UNKNOWN if scene_env == 'unknown_sim' else REAL_DIRS_UNKNOWN
    elif dataset_version == 'v3':
        if job.real:
            if scene_env == 'real_v3':
                if real_pos == 'out_of_grid':
                    dirs = REAL_DIRS_V3_OOG_ENVV3
                elif real_pos == 'v4':
                    dirs = REAL_DIRS_V4_ENVV3
                else:
                    dirs = REAL_DIRS_V3_ENVV3
            elif real_pos == 'out_of_grid':
                dirs = REAL_DIRS_NORIS_OOG if not use_ris else REAL_DIRS_V3_OOG
            elif real_pos == 'v4':
                dirs = REAL_DIRS_V4_NORIS  if not use_ris else REAL_DIRS_V4
            else:
                dirs = REAL_DIRS_V3_NORIS  if not use_ris else REAL_DIRS_V3
        else:
            dirs = SIM_DIRS_V3_NORIS if not use_ris else SIM_DIRS_V3
    else:
        dirs = REAL_DIRS if job.real else SIM_DIRS
    v = (N_SAMPLES - 1) if job.activity == 'IDLE' else job.variant_idx
    act_dir = Path(dirs[job.activity])
    if job.real:
        return BASE_DIR / act_dir / f'H_activity_v{v}.csv'
    else:
        return BASE_DIR / act_dir / f'rx{job.rx_idx}' / f'H_activity_v{v}.csv'


def execute_job(job, gpu, log_dir, python_bin, dataset_version='v2', real_pos='grid', use_ris=True, scene_env=None):
    out_file = _expected_output_file(job, dataset_version, real_pos, use_ris, scene_env)
    if out_file.exists():
        print(f'[SKIP] {job.name:<24} (già presente)')
        return True

    cmd = [python_bin, str(BASE_DIR / 'run_acquisition_parallel_v2.py'),
           '--worker', job.activity, str(job.rx_idx)]
    if job.variant_idx is not None:
        cmd += ['--variant-idx', str(job.variant_idx)]
    if job.real:
        cmd.append('--real')
    cmd += ['--dataset-version', dataset_version]
    cmd += ['--real-pos', real_pos]
    if not use_ris:
        cmd.append('--no-ris')
    if scene_env:
        cmd += ['--scene-env', scene_env]

    log_path = log_dir / f'{job.name}.log'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env,
                               stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = proc.returncode == 0
    status = 'OK  ' if ok else f'FAIL(rc={proc.returncode})'
    print(f'[{status}] {job.name:<24} gpu={gpu}  {dt / 60:.1f} min  log={log_path.name}')
    return ok


def run_job(job, gpu_queue, dep_future, log_dir, python_bin, dataset_version='v2', real_pos='grid', use_ris=True, scene_env=None):
    if dep_future is not None:
        if not dep_future.result():
            print(f'[SKIP] {job.name}: dipendenza {job.depends_on} fallita')
            return False
    gpu = gpu_queue.get()
    try:
        return execute_job(job, gpu, log_dir, python_bin, dataset_version, real_pos, use_ris, scene_env)
    finally:
        gpu_queue.put(gpu)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    # ── Modalità worker ───────────────────────────────────────────────────
    if '--worker' in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument('--worker',          required=True)
        p.add_argument('rx_idx',            type=int, nargs='?', default=0)
        p.add_argument('--variant-idx',     type=int, default=None)
        p.add_argument('--real',            action='store_true')
        p.add_argument('--dataset-version', default='v2')
        p.add_argument('--real-pos',        default='grid', choices=['grid', 'out_of_grid', 'v4'])
        p.add_argument('--no-ris',          action='store_true', default=False)
        p.add_argument('--scene-env',       default=None,
                        help='scena override (es. real_v3); default: real o sim in base a --real')
        args = p.parse_args()
        run_worker(args.worker.upper(), args.rx_idx, args.real, args.variant_idx,
                   args.dataset_version, args.real_pos, use_ris=not args.no_ris,
                   scene_env=args.scene_env)
        return

    # ── Modalità orchestratore ────────────────────────────────────────────
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sim',  action='store_true', help='genera posizioni RX simulate')
    p.add_argument('--real', action='store_true', help='genera la posizione RX reale (fissa)')
    p.add_argument('--activities', default=','.join(ACTIVITY_ORDER),
                    help='sottoinsieme attività, es. WALK,RUN')
    p.add_argument('--rx', default=None,
                    help='sottoinsieme indici RX simulati, es. 0,1,2; default: tutti')
    p.add_argument('--dataset-version', choices=['v2', 'v3'], default='v2',
                    help='v2: 4 RX fissi; v3: griglia 9×4 (36 RX)')
    p.add_argument('--real-pos', default='grid', choices=['grid', 'out_of_grid', 'v4'],
                    help='grid: rx8 [4,-1.5,1.0] → dataset_real_v3/; out_of_grid: [4,-1.0,1.0] → dataset_real_v3_out_of_grid/; v4: [0.5,0.0,1.0] → dataset_real_v4/')
    p.add_argument('--no-ris', action='store_true',
                    help='disabilita relay RIS: usa solo percorso diretto gNB→RX (→ dataset_*_noris/)')
    p.add_argument('--scene-env', default=None,
                    help='scena Sionna RT override, es. real_v3 (→ ttilab_furnished_v3); default: real o sim')
    p.add_argument('--gpus', default=None, help='es. 1,2,3 — default: auto-detect GPU libere')
    p.add_argument('--gpu-mem-threshold', type=int, default=8000,
                    help='MiB liberi minimi per GPU utilizzabile (default 8000)')
    p.add_argument('--jobs-per-gpu', type=int, default=3)
    p.add_argument('--skip-idle', action='store_true',
                    help='non rigenerare IDLE, assume già presente su disco')
    p.add_argument('--python-bin', default=DEFAULT_PYTHON_BIN)
    p.add_argument('--dry-run', action='store_true', help='stampa solo il piano dei job')
    args = p.parse_args()

    if not args.sim and not args.real:
        args.sim = True

    activities = [a.strip().upper() for a in args.activities.split(',') if a.strip()]
    if args.rx is not None:
        rx_indices = [int(r) for r in args.rx.split(',') if r.strip()]
    else:
        n = len(RX_POSITIONS_SIM_V3) if args.dataset_version == 'v3' else len(RX_POSITIONS_SIM)
        rx_indices = list(range(n))

    jobs = build_jobs(activities, rx_indices, args.sim, args.real,
                       regenerate_idle=not args.skip_idle)

    gpus = ([int(g) for g in args.gpus.split(',')]
            if args.gpus else detect_free_gpus(args.gpu_mem_threshold))

    print(f'GPU selezionate: {gpus}  (jobs-per-gpu={args.jobs_per_gpu}, '
          f'capacità parallela={len(gpus) * args.jobs_per_gpu})')
    print(f'Totale job: {len(jobs)}')
    for j in jobs:
        dep = f'  (dopo {j.depends_on})' if j.depends_on else ''
        print(f'  - {j.name:<26} activity={j.activity:<5} real={j.real}  rx_idx={j.rx_idx}{dep}')

    if args.dry_run:
        return

    log_dir = BASE_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)

    gpu_queue = queue.Queue()
    for g in gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_queue.put(g)

    use_ris = not args.no_ris
    t_start = time.time()
    futures = {}
    with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as executor:
        for job in jobs:
            dep_future = futures.get(job.depends_on) if job.depends_on else None
            futures[job.name] = executor.submit(
                run_job, job, gpu_queue, dep_future, log_dir, args.python_bin,
                args.dataset_version, args.real_pos, use_ris, args.scene_env)
        results = {name: f.result() for name, f in futures.items()}

    dt_total = time.time() - t_start
    ok  = [n for n, v in results.items() if v]
    bad = [n for n, v in results.items() if not v]
    print(f'\nCompletati: {len(ok)}/{len(results)}  in {dt_total / 60:.1f} min')
    if bad:
        print('Falliti:')
        for n in bad:
            print(f'  - {n}  (log: logs/{n}.log)')
        sys.exit(1)


if __name__ == '__main__':
    main()
