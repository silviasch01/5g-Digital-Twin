#!/usr/bin/env python3
"""
csi_har_benchmark_pt.py
Benchmark CNN e Bidirectional-LSTM per Human Activity Recognition (HAR)
su dati CSI (Channel State Information) simulati e reali.
Versione PyTorch multi-GPU con DistributedDataParallel (DDP).

Pipeline dataset (equivalente a dataset.ipynb + notebook HAR):
  1. Legge H_activity_v*.csv raw da dataset_v2/ / dataset_real_v2/
  2. Costruisce e salva dataset_har.csv (cached; rigenerato con --rebuild)
  3. Carica da dataset_har.csv per il training (stesso flusso dei notebook)

Scenari eseguiti per ogni architettura:
  1. Train su sim (70/30) → test su sim
  2. Zero-shot: modello addestrato su sim applicato al dataset reale
  3. Train su reale (70/30) → test su reale

Uso single-GPU (equivalente al benchmark single):
    python csi_har_benchmark_pt.py --features doppler --activities FALL,IDLE,WALK

Uso multi-GPU con DDP (torchrun gestisce il lancio dei processi):
    torchrun --nproc_per_node=4 csi_har_benchmark_pt.py --features doppler
    torchrun --nproc_per_node=2 csi_har_benchmark_pt.py --features all --arch cnn

    NOTA: --batch-size è per-GPU; con 4 GPU e --batch-size 32 il batch effettivo è 128.

GPU specifiche (senza torchrun):
    python csi_har_benchmark_pt.py --gpu 0
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 csi_har_benchmark_pt.py
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE — modifica qui per cambiare dataset o parametri di default
# ═══════════════════════════════════════════════════════════════════════════════

# ── Percorsi dataset (relativi alla directory dello script) ──────────────────
DIR_DATASET_SIM  = "dataset_v2"        # simulato: 4 posizioni RX
DIR_DATASET_REAL = "dataset_real_v2"   # reale:    posizione RX fissa

#DIR_DATASET_SIM  = "dataset"        # simulato: 4 posizioni RX
#DIR_DATASET_REAL = "dataset_real"   # reale:    posizione RX fissa



# ── Mapping attività → subdirectory ─────────────────────────────────────────
ACTIVITY_DIRS = {
    'FALL':  'fall_furnished',
    'IDLE':  'furnished_idle',
    'JUMP':  'jump_furnished',
    'RUN':   'run_furnished',
    'STAND': 'stand_furnished',
    'WALK':  'walk_furnished',
}

# ── Attività incluse di default ─────────────────────────────────────────────
# Per usarne un sottoinsieme: --activities FALL,RUN,STAND,WALK
DEFAULT_ACTIVITIES = ['FALL', 'IDLE', 'JUMP', 'RUN', 'STAND', 'WALK']

# ── Parametri dataset ────────────────────────────────────────────────────────
N_STEPS            = 40    # timestep per sample
N_SC               = 1272  # subcarrier per timestep
N_FEATURES_RAW     = 4     # canali raw: RE, IM, AMP, PHASE

# ── Feature Doppler attive ────────────────────────────────────────────────────
# Commentare una riga per escludere quella feature dal training.
# L'ordine determina l'indice nel tensore finale (dopo i 4 raw).
DOPPLER_FEATURES = [
    'dPHA_rel',           # dPHA / mean_t(|dPHA|) per SC — velocità di fase norm.
    'abs_dPHA_rel',       # |dPHA| / mean_t(|dPHA|) per SC — intensità Doppler norm.
    'dH_abs_rel',         # |ΔH| / mean_t(|ΔH|) per SC — variazione ampiezza norm.
    'rel_env_t',          # profilo temporale Doppler normalizzato (broadcast)
    'sign_frac_t',        # frazione SC con dPHA>0 — indicatore di direzione
    'kurt_env',           # kurtosis di env_t — alta per FALL (burst impulsivo)
    'fft_env_1',          # FFT di env_t bin 1 — periodicità bassa freq
    'fft_env_2',          # FFT di env_t bin 2 — cadenza del passo (WALK)
    'fft_env_3',          # FFT di env_t bin 3
    'fft_env_4',          # FFT di env_t bin 4
    'fft_env_5',          # FFT di env_t bin 5
    'sign_change_t',      # tasso inversioni direzione — WALK periodico, FALL basso
    'autocorr_env_1',     # autocorr rel_env a lag 1 — periodicità breve
    'autocorr_env_2',     # autocorr rel_env a lag 2
    'autocorr_env_3',     # autocorr rel_env a lag 3
    'delay_spread_norm',  # RMS delay spread da IFFT di H su SC — ricchezza multipath
    'sc_half_corr',       # correlazione Pearson tra semibande SC — coerenza freq.
    'delay_centroid_vel', # velocità del centroide di ritardo — dinamica multipath
    
    # ── Feature di energia assoluta — discriminano IDLE (H_d≈0) da attività bassa dinamica ──
    # Necessarie sempre: le feature Doppler sopra sono normalizzate sul proprio valor medio,
    # il che rende IDLE (H_d≈noise) e STAND (H_d piccolo) indistinguibili.
    # Con H_activity (subtract-mode none): IDLE ha |H_d|≈0, attività hanno |H_d|>0 → separazione netta.
    'log_amp_rms_t',      # log(1 + RMS_SC(|H_d|)) per step — cattura l'energia assoluta
    'amp_rms_t',          # RMS_SC(|H_d|) normalizzato per il massimo del campione — profilo
]

N_FEATURES_DOPPLER = N_FEATURES_RAW + len(DOPPLER_FEATURES)

# Subtract mode: none=H_activity | dt=DT prior puro | dt_ema=DT-initialized EMA
#              | combined=H_combined grezzo (H_s+H_d), nessuna sottrazione:
#                le feature Doppler cancellano H_s per costruzione (differenze
#                temporali), quindi non serve stimare H_s via DT/EMA.
# Controllato da --subtract-mode CLI; impostato dinamicamente in main().
APPLY_STATIC_SUBTRACTION = False  # default; sovrascritto in main() da --subtract-mode

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING E PRE-PROCESSING  (equivalente a dataset.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def _raw_csv_to_array(path: str) -> np.ndarray:
    """Legge H_activity_v{i}.csv → (N_STEPS, N_SC, N_FEATURES) float32."""
    steps = []
    with open(path) as fh:
        fh.readline()  # header: step;0;1;...;1271
        for line in fh:
            parts = line.rstrip('\n').split(';')
            sc_vals = [list(map(float, v.split(','))) for v in parts[1:]]
            steps.append(sc_vals)
    return np.array(steps, dtype=np.float32)


def _build_npz_from_raw(
    base_dir: str,
    activities: list[str],
    rx_indices: list[int] | None,
    npz_path: str,
) -> None:
    """Legge raw CSV e scrive direttamente il .npz — nessun file CSV intermedio.

    Pattern di file:
    - APPLY_STATIC_SUBTRACTION=False: H_activity_v*.csv (= H_d già pulita)
    - APPLY_STATIC_SUBTRACTION=True:
        IDLE     → H_activity_v*.csv  (= H_s grezzo; H_combined non esiste per IDLE,
                    dato che senza moto H_d≈0 e non c'è nulla da "combinare")
        non-IDLE → H_combined_v*.csv  (= H_s + H_d grezzo)
    Questo stesso pattern serve sia per dt/dt_ema (sottrazione H_s downstream)
    sia per combined (nessuna sottrazione: le feature Doppler cancellano H_s
    per costruzione via differenze temporali).
    """
    base = Path(base_dir)
    X_list, y_list, rx_list = [], [], []

    for act in activities:
        act_dir = base / ACTIVITY_DIRS[act]

        if rx_indices is not None:
            search_dirs = [(act_dir / f'rx{rx}', rx) for rx in rx_indices]
        else:
            rx_subdirs = sorted(d for d in act_dir.glob('rx*') if d.is_dir())
            if rx_subdirs:
                search_dirs = [(d, int(d.name[2:])) for d in rx_subdirs]
            else:
                search_dirs = [(act_dir, 0)]

        pattern = ('H_combined_v*.csv'
                   if APPLY_STATIC_SUBTRACTION and act != 'IDLE'
                   else 'H_activity_v*.csv')

        n_act = 0
        for dir_path, rx_id in search_dirs:
            for raw_path in sorted(dir_path.glob(pattern)):
                H = _raw_csv_to_array(str(raw_path))
                if H.shape != (N_STEPS, N_SC, N_FEATURES_RAW):
                    print(f'  [WARN] shape inattesa {H.shape} — {raw_path}', file=sys.stderr)
                    continue
                X_list.append(H)
                y_list.append(act)
                rx_list.append(rx_id)
                n_act += 1
            if n_act == 0:
                print(f'  [WARN] nessun file {pattern} in {dir_path}', file=sys.stderr)

        print(f'  {act} [{pattern}]: {n_act} sample')

    X  = np.stack(X_list)
    y  = np.array(y_list)
    rx = np.array(rx_list, dtype=np.int32)
    np.savez_compressed(npz_path, X=X, y=y, rx_ids=rx)
    print(f'  → {npz_path}  shape={X.shape}')


def load_dataset(
    base_dir: str,
    activities: list[str],
    rx_indices: list[int] | None = None,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carica dataset da .npz; se non esiste lo costruisce direttamente dai raw CSV."""
    import time
    _suffix  = 'combined' if APPLY_STATIC_SUBTRACTION else 'activity'
    npz_path = Path(base_dir) / f'dataset_har_{_suffix}.npz'

    if rebuild and npz_path.exists():
        npz_path.unlink()

    if npz_path.exists():
        data    = np.load(str(npz_path), allow_pickle=True)
        X, y, rx_ids = data['X'], data['y'], data['rx_ids']
        missing = set(activities) - set(np.unique(y))
        if missing:
            print(f'  [INFO] attività mancanti nel .npz ({missing}) — ricostruzione...')
            npz_path.unlink()

    if not npz_path.exists():
        print(f'  Costruzione {npz_path.name} dai raw CSV...')
        t0 = time.time()
        _build_npz_from_raw(base_dir, activities, rx_indices, str(npz_path))
        print(f'  Build completato in {time.time()-t0:.1f}s')
        data    = np.load(str(npz_path), allow_pickle=True)
        X, y, rx_ids = data['X'], data['y'], data['rx_ids']

    t0   = time.time()
    mask = np.isin(y, activities)
    X, y, rx_ids = X[mask], y[mask], rx_ids[mask]
    if rx_indices is not None:
        rx_mask = np.isin(rx_ids, rx_indices)
        X, y, rx_ids = X[rx_mask], y[rx_mask], rx_ids[rx_mask]
    print(f'  Caricamento da {npz_path.name} in {time.time()-t0:.1f}s')
    print(f'  X shape : {X.shape}   classi: {np.unique(y)}')
    return X, y, rx_ids


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING DOPPLER  (da doppler_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def subtract_static_prior(
    X: np.ndarray,
    y: np.ndarray,
    rx_ids: np.ndarray,
    H_s_map: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Sottrae il prior statico H_s per-receiver dal canale raw.

    H_s_map: dict {rx_id: (SC,) complex} — prior per ogni receiver.
    Se non fornito, viene stimato dai campioni IDLE di ogni receiver.
    Restituisce (X_d, H_s_map):
      X_d    — (N, T, SC, 4) con H_d = H - H_s^rx  (RE, IM, AMP, PHASE aggiornati)
      H_s_map — dict usato (da passare a REAL in zero-shot)
    """
    H_cx = X[..., 0].astype(np.float64) + 1j * X[..., 1].astype(np.float64)

    if H_s_map is None:
        H_s_map = {}
        for rx_id in np.unique(rx_ids):
            mask = (rx_ids == rx_id) & (y == 'IDLE')
            if not mask.any():
                raise ValueError(
                    f'Nessun campione IDLE per rx{rx_id}. '
                    'Includi IDLE nelle attività o passa H_s_map esplicitamente.'
                )
            H_s_map[rx_id] = H_cx[mask].mean(axis=(0, 1))
            print(f'  H_s^rx{rx_id} stimato da {mask.sum()} campioni IDLE  '
                  f'|H_s| medio={np.abs(H_s_map[rx_id]).mean():.4f}')

    H_d = H_cx.copy()
    for rx_id, H_s in H_s_map.items():
        m = (rx_ids == rx_id)
        H_d[m] = H_cx[m] - H_s[None, None, :]

    X_d = X.copy().astype(np.float32)
    X_d[..., 0] = H_d.real.astype(np.float32)
    X_d[..., 1] = H_d.imag.astype(np.float32)
    X_d[..., 2] = np.abs(H_d).astype(np.float32)
    X_d[..., 3] = np.angle(H_d).astype(np.float32)

    return X_d, H_s_map


EMA_ALPHA = 0.80   # smoothing factor (= alpha usato durante l'acquisizione)


def apply_dt_ema(
    X: np.ndarray,
    y: np.ndarray,
    rx_ids: np.ndarray,
    H_s_map: dict | None = None,
    alpha: float = EMA_ALPHA,
) -> tuple[np.ndarray, dict]:
    """DT-initialized EMA per-receiver.

    H_bg[0] = H_s^DT  (prior DT come warm-start)
    H_bg[t] = alpha * H_bg[t-1] + (1-alpha) * H[t]
    H_d[t]  = H[t] - H_bg[t]

    H_s_map=None  → stima H_s^DT dai campioni IDLE (SIM).
    H_s_map=<dict> → usa H_s^sim come init per REAL (zero-shot): l'EMA poi
                      adatta progressivamente al canale statico reale.
    Vettorizzato su N e SC; solo T=40 iterazioni Python.
    """
    H_cx = X[..., 0].astype(np.float64) + 1j * X[..., 1].astype(np.float64)  # (N, T, SC)

    if H_s_map is None:
        H_s_map = {}
        for rx_id in np.unique(rx_ids):
            mask = (rx_ids == rx_id) & (y == 'IDLE')
            if not mask.any():
                raise ValueError(
                    f'Nessun campione IDLE per rx{rx_id}. '
                    'Includi IDLE nelle attività o passa H_s_map esplicitamente.'
                )
            H_s_map[rx_id] = H_cx[mask].mean(axis=(0, 1))
            print(f'  H_s^DT rx{rx_id} da {mask.sum()} campioni IDLE  '
                  f'|H_s| medio={np.abs(H_s_map[rx_id]).mean():.4f}')

    X_d = np.zeros_like(X, dtype=np.float32)

    for rx_id, H_s in H_s_map.items():
        m = (rx_ids == rx_id)
        if not m.any():
            continue
        H_cx_rx = H_cx[m]                                       # (N_rx, T, SC)
        N_rx    = H_cx_rx.shape[0]
        H_bg    = np.tile(H_s[np.newaxis, :], (N_rx, 1))       # (N_rx, SC) init da DT
        for t in range(H_cx_rx.shape[1]):
            H_bg             = alpha * H_bg + (1 - alpha) * H_cx_rx[:, t, :]
            H_d_t            = H_cx_rx[:, t, :] - H_bg
            X_d[m, t, :, 0] = H_d_t.real.astype(np.float32)
            X_d[m, t, :, 1] = H_d_t.imag.astype(np.float32)
            X_d[m, t, :, 2] = np.abs(H_d_t).astype(np.float32)
            X_d[m, t, :, 3] = np.angle(H_d_t).astype(np.float32)

    return X_d, H_s_map


def augment_features(X: np.ndarray, demean_amp: bool = False) -> np.ndarray:
    """Arricchisce l'input con feature Doppler scale-invarianti per il sim-to-real transfer.

    Input : (N, N_STEPS, N_SC, 4)  — canali raw: RE, IM, AMP, PHA
    demean_amp: se True, sottrae la media temporale per-SC di |H| prima di
        calcolare le feature di energia assoluta [22][23] (Gruppo D). Necessario
        con --subtract-mode combined, dove X = H_s+H_d grezzo: senza demean,
        |H| è dominato da H_s (dipendente da path-loss/posizione RX, non dal
        moto), e le feature [22][23] finiscono per codificare la posizione del
        ricevitore anziché l'attività. La media è calcolata sulla finestra
        stessa (T=40 step, H_s costante per costruzione), quindi non richiede
        una stima di H_s via DT/EMA — stesso principio delle feature Doppler
        differenziali [4]-[21], applicato qui via rimozione della componente DC
        invece che via differenza tra frame successivi.
    Output: (N, N_STEPS, N_SC, 24)  — raw + 20 canali Doppler:
        [4]  dPHA_rel      — dPHA / mean_t(|dPHA|) per SC (scale-invariante)
        [5]  abs_dPHA_rel  — |dPHA| / mean_t(|dPHA|) per SC (scale-invariante)
        [6]  dH_abs_rel    — |ΔH| / mean_t(|ΔH|) per SC (scale-invariante)
        [7]  rel_env_t     — profilo temporale Doppler normalizzato (broadcast)
        [8]  sign_frac_t   — frazione SC con dPHA>0
        [9]  kurt_env      — kurtosis temporale di env_t (broadcast): alta per FALL
        [10..14] fft_env[1..5] — FFT normalizzata di env_t (broadcast)
        [15] sign_change_t — frazione SC con inversione di segno dPHA (broadcast)
        [16..18] autocorr_env[1..3] — autocorrelazione di rel_env_t a lag 1,2,3
        [19] delay_spread_norm_t   — RMS delay spread da IFFT di H lungo SC (broadcast)
        [20] sc_half_corr_t        — correlazione Pearson ampiezza tra semibande SC
        [21] delay_centroid_vel_t  — velocità del centroide di ritardo (broadcast)
        [22] log_amp_rms_t  — log(1+RMS_SC(|H|)) per step; scale-variant, discrimina IDLE
        [23] amp_rms_t      — RMS_SC(|H|)/max normalizzato per il campione
    """
    N, T, S, _ = X.shape

    H_cx     = X[..., 0].astype(np.float32) + 1j * X[..., 1].astype(np.float32)
    PHA_uw   = np.unwrap(X[..., 3].astype(np.float32), axis=1)

    _dPHA    = np.diff(PHA_uw, axis=1)
    dPHA     = np.concatenate([np.zeros((N, 1, S), np.float32), _dPHA], axis=1)
    abs_dPHA = np.abs(dPHA)

    _dH      = np.abs(np.diff(H_cx, axis=1)).astype(np.float32)
    dH_abs   = np.concatenate([np.zeros((N, 1, S), np.float32), _dH], axis=1)

    # Normalizzazione per-SC lungo l'asse temporale: rimuove il guadagno assoluto
    # per subcarrier (dipendente da hardware/geometria), mantiene il profilo temporale
    dPHA_rel     = dPHA     / (abs_dPHA.mean(axis=1, keepdims=True) + 1e-6)
    abs_dPHA_rel = abs_dPHA / (abs_dPHA.mean(axis=1, keepdims=True) + 1e-6)
    dH_abs_rel   = dH_abs   / (dH_abs.mean(axis=1, keepdims=True) + 1e-6)

    # rel_env_t: profilo temporale Doppler normalizzato — scale-invariante
    env_t_seq = abs_dPHA.mean(axis=2)                                          # (N, T)
    mean_env  = env_t_seq.mean(axis=1, keepdims=True) + 1e-6
    rel_env_t = np.broadcast_to(
        (env_t_seq / mean_env)[:, :, None], (N, T, S)
    ).copy().astype(np.float32)

    # sign_frac_t: frazione di SC con dPHA>0 — indicatore di direzione
    sign_frac_bc = np.broadcast_to(
        (dPHA > 0).astype(np.float32).mean(axis=2)[:, :, None], (N, T, S)
    ).copy().astype(np.float32)

    # kurtosis temporale di env_t — scale-invariante, alta per FALL (burst)
    mu_e    = env_t_seq.mean(axis=1, keepdims=True)
    sig_e   = env_t_seq.std(axis=1, keepdims=True) + 1e-6
    kurt    = ((((env_t_seq - mu_e) / sig_e) ** 4).mean(axis=1) - 3)          # (N,)
    kurt_bc = np.broadcast_to(kurt[:, None, None], (N, T, S)).copy().astype(np.float32)

    # FFT di env_t — bin 1..5 catturano periodicità del passo (domain-invariante)
    fft_mag  = np.abs(np.fft.rfft(env_t_seq, axis=1))[:, 1:6]                 # (N, 5)
    fft_norm = fft_mag / (fft_mag.max(axis=1, keepdims=True) + 1e-6)
    fft_channels = [
        np.broadcast_to(fft_norm[:, k, None, None], (N, T, S)).copy().astype(np.float32)
        for k in range(5)
    ]

    # sign_change_t: tasso di inversione di direzione Doppler per step — scale-invariante
    # WALK: inversioni regolari al ritmo del passo; FALL: basso; STAND: random
    dPHA_prev      = np.concatenate([np.zeros((N, 1, S), np.float32), dPHA[:, :-1, :]], axis=1)
    sign_change    = (dPHA * dPHA_prev < 0).astype(np.float32).mean(axis=2)   # (N, T)
    sign_change_bc = np.broadcast_to(
        sign_change[:, :, None], (N, T, S)
    ).copy().astype(np.float32)

    # autocorrelazione di rel_env_t a lag 1,2,3 — scale-invariante
    # WALK: alta (periodica); FALL: negativa ai lag brevi (burst poi calo)
    rel_env_seq   = rel_env_t[:, :, 0]                                         # (N, T)
    var_env       = rel_env_seq.var(axis=1) + 1e-6                             # (N,)
    autocorr_channels = []
    for lag in (1, 2, 3):
        r = (rel_env_seq[:, :-lag] * rel_env_seq[:, lag:]).mean(axis=1) / var_env
        autocorr_channels.append(
            np.broadcast_to(r[:, None, None], (N, T, S)).copy().astype(np.float32)
        )

    # ── Features frequenziali cross-SC (sfruttano la struttura spettrale di H) ─

    # delay_spread_norm_t: RMS delay spread normalizzato — cattura ricchezza di multipath
    # IFFT di H lungo axis SC → power delay profile; spread aumenta con corpo in scatter
    H_delay    = np.fft.ifft(H_cx, axis=2)                                     # (N, T, S)
    delay_pow  = np.abs(H_delay).astype(np.float32) ** 2                       # (N, T, S)
    total_pow  = delay_pow.sum(axis=2, keepdims=True) + 1e-6
    bins_norm  = (np.arange(S, dtype=np.float32) / S)[None, None, :]           # (1, 1, S)
    centroid   = (delay_pow / total_pow * bins_norm).sum(axis=2)               # (N, T)
    spread     = np.sqrt(
        ((delay_pow / total_pow) * (bins_norm - centroid[:, :, None]) ** 2).sum(axis=2)
    )                                                                           # (N, T) scale-inv
    spread_rel = spread / (spread.mean(axis=1, keepdims=True) + 1e-6)
    spread_bc  = np.broadcast_to(spread_rel[:, :, None], (N, T, S)).copy().astype(np.float32)

    # sc_half_corr_t: correlazione Pearson tra ampiezza della prima e seconda metà SC
    # Canale flat (1 path dominante) → corr ≈ 1; scatter corporeo → corr scende
    AMP  = X[..., 2].astype(np.float32)                                        # (N, T, S)
    half = S // 2
    A1   = AMP[:, :, :half] - AMP[:, :, :half].mean(axis=2, keepdims=True)
    A2   = AMP[:, :, half:] - AMP[:, :, half:].mean(axis=2, keepdims=True)
    sc_corr = (A1 * A2).sum(axis=2) / (
        np.sqrt((A1 ** 2).sum(axis=2) * (A2 ** 2).sum(axis=2)) + 1e-6
    )                                                                           # (N, T)
    sc_corr_bc = np.broadcast_to(sc_corr[:, :, None], (N, T, S)).copy().astype(np.float32)

    # delay_centroid_vel_t: velocità del centroide di ritardo — cattura dinamica del multipath
    # FALL: velocità non nulla poi zero; WALK: periodica; IDLE: ~zero
    d_centroid   = np.diff(centroid, axis=1)                                   # (N, T-1)
    d_centroid   = np.concatenate(
        [np.zeros((N, 1), np.float32), d_centroid.astype(np.float32)], axis=1
    )                                                                           # (N, T)
    vel_rel      = d_centroid / (np.abs(d_centroid).mean(axis=1, keepdims=True) + 1e-6)
    vel_bc       = np.broadcast_to(vel_rel[:, :, None], (N, T, S)).copy().astype(np.float32)

    # ── Feature di energia assoluta (scale-variant) ──────────────────────────
    # Nota: queste feature non sono normalizzate intra-campione — il loro valore
    # assoluto è informativo. Dopo StandardScaler (fit su sim), IDLE si colloca
    # sistematicamente più in basso rispetto alle attività con H_d non trascurabile.
    AMP_raw = X[..., 2].astype(np.float32)                                  # (N, T, S)
    if demean_amp:
        # Rimuove la componente statica per-SC (H_s, costante sulla finestra):
        # isola la fluttuazione temporale dovuta al moto senza stimare H_s.
        AMP_raw = AMP_raw - AMP_raw.mean(axis=1, keepdims=True)
    amp_rms_t_seq = np.sqrt((AMP_raw ** 2).mean(axis=2))                    # (N, T)
    # log(1+x): comprime la scala e mantiene la separazione IDLE vs non-IDLE
    log_amp_seq = np.log1p(amp_rms_t_seq).astype(np.float32)               # (N, T)
    log_amp_bc  = np.broadcast_to(log_amp_seq[:, :, None], (N, T, S)).copy().astype(np.float32)
    # profilo normalizzato sul max del campione: preserva la forma temporale relativa
    amp_rms_max = amp_rms_t_seq.max(axis=1, keepdims=True) + 1e-6
    amp_rms_rel = (amp_rms_t_seq / amp_rms_max).astype(np.float32)         # (N, T)
    amp_rms_bc  = np.broadcast_to(amp_rms_rel[:, :, None], (N, T, S)).copy().astype(np.float32)

    _all = {
        'dPHA_rel':           dPHA_rel,
        'abs_dPHA_rel':       abs_dPHA_rel,
        'dH_abs_rel':         dH_abs_rel,
        'rel_env_t':          rel_env_t,
        'sign_frac_t':        sign_frac_bc,
        'kurt_env':           kurt_bc,
        'fft_env_1':          fft_channels[0],
        'fft_env_2':          fft_channels[1],
        'fft_env_3':          fft_channels[2],
        'fft_env_4':          fft_channels[3],
        'fft_env_5':          fft_channels[4],
        'sign_change_t':      sign_change_bc,
        'autocorr_env_1':     autocorr_channels[0],
        'autocorr_env_2':     autocorr_channels[1],
        'autocorr_env_3':     autocorr_channels[2],
        'delay_spread_norm':  spread_bc,
        'sc_half_corr':       sc_corr_bc,
        'delay_centroid_vel': vel_bc,
        'log_amp_rms_t':      log_amp_bc,
        'amp_rms_t':          amp_rms_bc,
    }
    return np.concatenate(
        [X.astype(np.float32)] + [_all[k][..., None] for k in DOPPLER_FEATURES],
        axis=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITETTURE  (single-GPU, nessuna strategy)
# ═══════════════════════════════════════════════════════════════════════════════

class CNN(nn.Module):
    """CNN 2D per HAR su CSI. Input atteso: (N, T, SC, F); il forward permuta a (N, F, T, SC)."""

    def __init__(self, n_feat: int, num_classes: int):
        super().__init__()

        def _block(in_c, out_c, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation),
                nn.LeakyReLU(0.1),
                nn.BatchNorm2d(out_c),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            _block(n_feat, 64),
            _block(64,  128),
            _block(128, 256),
            _block(256, 256),
            _block(256, 512, dilation=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)          # (N, T, SC, F) → (N, F, T, SC)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class BiLSTM(nn.Module):
    """Bidirectional LSTM per HAR su CSI. Input atteso: (N, T, SC*F)."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.proj  = nn.Linear(in_features, 256)
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.bn1   = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(256,  64, batch_first=True, bidirectional=True)
        self.bn2   = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(x))                               # (N, T, 256)
        x, _ = self.lstm1(x)                                   # (N, T, 256)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)     # BN su dim feature
        x = self.drop1(x)
        x, _ = self.lstm2(x)                                   # (N, T, 128)
        x = x[:, -1, :]                                        # ultimo timestep
        x = self.bn2(x)
        x = self.drop2(x)
        return self.classifier(x)


class Transformer(nn.Module):
    """Transformer encoder per HAR su CSI. Input: (N, T, SC*F)."""

    def __init__(self, in_features: int, num_classes: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(x))    # (N, T, d_model)
        x = self.encoder(x)          # (N, T, d_model)
        x = x.mean(dim=1)            # mean pooling su T
        return self.classifier(x)


def build_cnn(n_feat: int, num_classes: int) -> tuple:
    """Restituisce (model, lr) per la CNN."""
    return CNN(n_feat, num_classes), 3e-4


def build_lstm(in_features: int, num_classes: int) -> tuple:
    """Restituisce (model, lr) per il BiLSTM."""
    return BiLSTM(in_features, num_classes), 5e-4


def build_transformer(in_features: int, num_classes: int) -> tuple:
    """Restituisce (model, lr) per il Transformer encoder."""
    return Transformer(in_features, num_classes), 3e-4


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(
    model: nn.Module,
    lr: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: str,
    epochs: int = 200,
    batch_size: int = 4,
    device: torch.device = torch.device('cpu'),
    arch: str = '',
    n_feat: int = 0,
    num_classes: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Addestra model e salva il best checkpoint. Supporta DDP se world_size > 1.

    Con DDP: batch_size è per-GPU, il batch effettivo = batch_size * world_size.
    Solo rank 0 salva checkpoint, stampa progress e decide l'early stopping;
    il segnale di stop viene broadcastato agli altri rank.
    """
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    base_model = model.module if world_size > 1 else model

    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=device.type == 'cuda')

    Xv    = torch.from_numpy(X_val).float()   # kept on CPU, moved in mini-batches
    yv    = torch.from_numpy(y_val).long()

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.7, patience=3, min_lr=1e-5
    )

    best_acc     = 0.0
    patience_ctr = 0
    PATIENCE     = 25

    for epoch in range(1, epochs + 1):
        if world_size > 1:
            sampler.set_epoch(epoch)  # garantisce shuffle diverso per ogni epoch

        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        # Tutti i rank calcolano le metriche di val (stesso risultato dopo DDP sync)
        model.eval()
        with torch.no_grad():
            _parts = [base_model(Xv[i:i+batch_size].to(device)).cpu()
                      for i in range(0, len(Xv), batch_size)]
            logits   = torch.cat(_parts)
            val_loss = F.cross_entropy(logits, yv).item()
            val_acc  = (logits.argmax(1) == yv).float().mean().item()

        scheduler.step(val_loss)
        lr_now = opt.param_groups[0]['lr']

        # Solo rank 0 salva e stampa; poi broadcasta la decisione di stop
        if rank == 0:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {'arch': arch, 'n_feat': n_feat, 'num_classes': num_classes,
                     'state_dict': base_model.state_dict()},
                    save_path,
                )
                patience_ctr = 0
                marker = ' ← best'
            else:
                patience_ctr += 1
                marker = f' (patience {patience_ctr}/{PATIENCE})'

            if epoch % 10 == 0 or epoch == 1:
                print(f'  Epoch {epoch:4d}/{epochs}  '
                      f'loss={total_loss/len(X_train):.4f}  val_loss={val_loss:.4f}  '
                      f'val_acc={val_acc*100:.1f}%  lr={lr_now:.1e}{marker}')

        # Sincronizza il segnale di early stopping da rank 0 a tutti gli altri
        if world_size > 1:
            stop = torch.tensor([patience_ctr >= PATIENCE], dtype=torch.bool, device=device)
            dist.broadcast(stop, src=0)
            if stop.item():
                if rank == 0:
                    print(f'  Early stopping a epoch {epoch}  (best val_acc={best_acc*100:.1f}%)')
                break
        elif patience_ctr >= PATIENCE:
            print(f'  Early stopping a epoch {epoch}  (best val_acc={best_acc*100:.1f}%)')
            break

    if rank == 0:
        print(f'  Best val_acc: {best_acc*100:.1f}%  → {save_path}')

    # Barriera: assicura che rank 0 abbia scritto il checkpoint prima che gli altri
    # procedano a leggere/valutare il modello salvato.
    if world_size > 1:
        dist.barrier()


# ═══════════════════════════════════════════════════════════════════════════════
# VALUTAZIONE E PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model_from_ckpt(path: str, device: torch.device) -> nn.Module:
    """Ricostruisce il modello dal checkpoint salvato da train_and_save."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch, n_feat, num_classes = ckpt['arch'], ckpt['n_feat'], ckpt['num_classes']
    if arch == 'CNN':
        model = CNN(n_feat, num_classes)
    elif arch == 'TRANSFORMER':
        model = Transformer(N_SC * n_feat, num_classes)
    else:
        model = BiLSTM(N_SC * n_feat, num_classes)
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_enc: LabelEncoder,
    title: str,
    save_prefix: str,
    out_dir: str,
    accuracy: float,
) -> None:
    """Plotta e salva la confusion matrix normalizzata."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    class_labels = label_enc.classes_
    fig, ax = plt.subplots(figsize=(max(6, len(class_labels)), max(5, len(class_labels) - 1)))
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={'size': 12},
        vmin=0, vmax=1,
        ax=ax,
    )
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    #ax.set_title(f'{title}  acc={accuracy * 100:.1f}%', fontsize=13)
    ax.tick_params(axis='x', labelsize=11, rotation=30)
    ax.tick_params(axis='y', labelsize=11, rotation=0)
    fig.tight_layout()
    pdf_path = os.path.join(out_dir, f'{save_prefix}.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura salvata → {pdf_path}')


def evaluate_and_plot(
    model_path: str,
    X: np.ndarray,
    y_enc: np.ndarray,
    label_enc: LabelEncoder,
    title: str,
    save_prefix: str,
    out_dir: str,
    device: torch.device = torch.device('cpu'),
) -> float:
    """Carica il best model PyTorch, predice su X, plotta e salva la confusion matrix."""
    model = _load_model_from_ckpt(model_path, device)
    Xt = torch.from_numpy(X).float()
    _bs = 32
    with torch.no_grad():
        parts = [model(Xt[i:i+_bs].to(device)).cpu() for i in range(0, len(Xt), _bs)]
        logits = torch.cat(parts)
    y_pred   = logits.argmax(1).numpy()
    accuracy = float(np.mean(y_pred == y_enc))
    print(f'  Accuratezza [{title}]: {accuracy * 100:.1f}%')
    _plot_confusion_matrix(y_enc, y_pred, label_enc, title, save_prefix, out_dir, accuracy)
    return accuracy


def _print_section(title: str) -> None:
    print(f'\n{"═" * 70}')
    print(f'  {title}')
    print(f'{"═" * 70}')


def _print_class_dist(prefix: str, le: LabelEncoder, y_enc: np.ndarray) -> None:
    classes, counts = np.unique(y_enc, return_counts=True)
    dist = '  '.join(f'{le.classes_[c]}:{n}' for c, n in zip(classes, counts))
    print(f'{prefix}  {dist}')


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARI
# ═══════════════════════════════════════════════════════════════════════════════

def run_architecture(
    arch: str,
    X_sim: np.ndarray,
    y_sim: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    label_enc_sim: LabelEncoder,
    figs_dir: str,
    models_dir: str,
    epochs: int,
    batch_size: int,
    features: str = 'raw',
    eval_only: bool = False,
    enable_real_training: bool = True,
    device: torch.device = torch.device('cpu'),
    rank: int = 0,
    world_size: int = 1,
    subtract_tag: str = '',
) -> dict[str, float]:
    """Esegue i 3 scenari per un'architettura (CNN o LSTM).

    figs_dir   → dove vengono salvati i PDF delle confusion matrix
    models_dir → dove vengono salvati/cercati i file .pt

    eval_only=True: salta il training e genera solo le confusion matrix dai
    modelli già presenti in models_dir. Il test set è ricostruito con lo
    stesso seed del training (random_state=42) per valutare sugli stessi campioni.
    """
    tag        = f'{arch.lower()}_{features}' + (f'_{subtract_tag}' if subtract_tag else '')
    feat_label = {'raw': 'raw', 'doppler': 'Doppler only', 'all': 'raw+Doppler'}[features]
    results    = {}

    n_feat = X_sim.shape[-1]

    def maybe_reshape(X):
        if arch in ('LSTM', 'TRANSFORMER'):
            return X.reshape(len(X), N_STEPS, N_SC * n_feat)
        return X

    num_classes = len(label_enc_sim.classes_)

    # ── Scenario 1: train su sim → test su sim ───────────────────────────────
    if rank == 0:
        _print_section(f'{arch} — Scenario 1: train su sim / test su sim')

    y_sim_enc = label_enc_sim.transform(y_sim)

    idx = np.arange(len(X_sim))
    np.random.seed(42)
    np.random.shuffle(idx)
    X_sim_sh = X_sim[idx]
    y_sim_sh  = y_sim_enc[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sim_sh, y_sim_sh, test_size=0.3, random_state=42,
    )
    X_tr_r, X_te_r = maybe_reshape(X_tr), maybe_reshape(X_te)

    if rank == 0:
        print(f'\n  Test: {X_te_r.shape}')
        _print_class_dist('  Test ', label_enc_sim, y_te)

    model_sim_path = os.path.join(models_dir, f'{tag}_sim.pt')
    if eval_only:
        if rank == 0:
            if not os.path.exists(model_sim_path):
                print(f'  [SKIP] model not found: {model_sim_path}')
            else:
                results['sim_test'] = evaluate_and_plot(
                    model_sim_path, X_te_r, y_te,
                    label_enc_sim,
                    title=f'{arch} [{feat_label}] — sim test',
                    save_prefix=f'cm_{tag}_train_sim_test_sim',
                    out_dir=figs_dir, device=device,
                )
    else:
        in_feat = n_feat if arch == 'CNN' else N_SC * n_feat
        if rank == 0:
            print(f'  Train: {X_tr_r.shape}')
            _print_class_dist('  Train', label_enc_sim, y_tr)
        if arch == 'CNN':
            model, lr = build_cnn(n_feat, num_classes)
        elif arch == 'TRANSFORMER':
            model, lr = build_transformer(in_feat, num_classes)
        else:
            model, lr = build_lstm(in_feat, num_classes)
        if rank == 0:
            print(model)
        train_and_save(model, lr, X_tr_r, y_tr, X_te_r, y_te, model_sim_path,
                       epochs, batch_size, device=device,
                       arch=arch, n_feat=n_feat, num_classes=num_classes,
                       rank=rank, world_size=world_size)
        if rank == 0:
            results['sim_test'] = evaluate_and_plot(
                model_sim_path, X_te_r, y_te,
                label_enc_sim,
                title=f'{arch} [{feat_label}] — sim test',
                save_prefix=f'cm_{tag}_train_sim_test_sim',
                out_dir=figs_dir, device=device,
            )
        if world_size > 1:
            dist.barrier()

    # ── Scenario 2: zero-shot su reale ──────────────────────────────────────
    if rank == 0:
        _print_section(f'{arch} — Scenario 2: zero-shot su reale')

    y_real_enc_sim = label_enc_sim.transform(y_real)
    X_real_r = maybe_reshape(X_real)

    if rank == 0:
        if not os.path.exists(model_sim_path):
            print(f'  [SKIP] model not found: {model_sim_path}')
        else:
            results['zeroshot_real'] = evaluate_and_plot(
                model_sim_path, X_real_r, y_real_enc_sim,
                label_enc_sim,
                title=f'{arch} [{feat_label}] — zero-shot real',
                save_prefix=f'cm_{tag}_train_sim_test_real',
                out_dir=figs_dir, device=device,
            )

    # ── Scenario 3: train su reale → test su reale ──────────────────────────
    if not enable_real_training:
        if rank == 0:
            print('\n  [SKIP] Scenario 3 disabilitato (usa --train-real per abilitarlo)')
        return results

    if rank == 0:
        _print_section(f'{arch} — Scenario 3: train su reale / test su reale')

    le_real    = LabelEncoder().fit(y_real)
    y_real_enc = le_real.transform(y_real)

    idx_r = np.arange(len(X_real))
    np.random.seed(42)
    np.random.shuffle(idx_r)
    X_real_sh = X_real[idx_r]
    y_real_sh  = y_real_enc[idx_r]

    X_tr_r2, X_te_r2, y_tr_r2, y_te_r2 = train_test_split(
        X_real_sh, y_real_sh, test_size=0.3, random_state=42,
    )
    X_tr_r2, X_te_r2 = maybe_reshape(X_tr_r2), maybe_reshape(X_te_r2)

    nc_real = len(le_real.classes_)
    model_real_path = os.path.join(models_dir, f'{tag}_real.pt')
    if eval_only:
        if rank == 0:
            print(f'\n  Test: {X_te_r2.shape}')
            _print_class_dist('  Test ', le_real, y_te_r2)
            if not os.path.exists(model_real_path):
                print(f'  [SKIP] model not found: {model_real_path}')
            else:
                results['real_trained'] = evaluate_and_plot(
                    model_real_path, X_te_r2, y_te_r2,
                    le_real,
                    title=f'{arch} [{feat_label}] — trained on real',
                    save_prefix=f'cm_{tag}_train_real_test_real',
                    out_dir=figs_dir, device=device,
                )
    else:
        in_feat = n_feat if arch == 'CNN' else N_SC * n_feat
        if rank == 0:
            print(f'\n  Train: {X_tr_r2.shape}  |  Test: {X_te_r2.shape}')
            _print_class_dist('  Train', le_real, y_tr_r2)
            _print_class_dist('  Test ', le_real, y_te_r2)
        if arch == 'CNN':
            model_r, lr = build_cnn(n_feat, nc_real)
        elif arch == 'TRANSFORMER':
            model_r, lr = build_transformer(in_feat, nc_real)
        else:
            model_r, lr = build_lstm(in_feat, nc_real)
        train_and_save(model_r, lr, X_tr_r2, y_tr_r2, X_te_r2, y_te_r2,
                       model_real_path, epochs, batch_size, device=device,
                       arch=arch, n_feat=n_feat, num_classes=nc_real,
                       rank=rank, world_size=world_size)
        if rank == 0:
            results['real_trained'] = evaluate_and_plot(
                model_real_path, X_te_r2, y_te_r2,
                le_real,
                title=f'{arch} [{feat_label}] — trained on real',
                save_prefix=f'cm_{tag}_train_real_test_real',
                out_dir=figs_dir, device=device,
            )
        if world_size > 1:
            dist.barrier()

    return results


def _sklearn_featurize(X: np.ndarray) -> np.ndarray:
    """(N, T, SC, F) → (N, T * 2 * F): mean e std su SC per preservare la dinamica temporale."""
    mu  = X.mean(axis=2)   # (N, T, F)
    std = X.std(axis=2)    # (N, T, F)
    return np.concatenate([mu, std], axis=2).reshape(len(X), -1)


def run_sklearn_arch(
    arch: str,
    X_sim: np.ndarray,
    y_sim: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    label_enc_sim: LabelEncoder,
    figs_dir: str,
    models_dir: str,
    features: str = 'doppler',
    enable_real_training: bool = True,
    rank: int = 0,
    subtract_tag: str = '',
) -> dict[str, float]:
    """Esegue i 3 scenari per RF o SVM (sklearn, single-process, rank-0 only)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import joblib

    if rank != 0:
        return {}

    tag        = f'{arch.lower()}_{features}' + (f'_{subtract_tag}' if subtract_tag else '')
    feat_label = {'raw': 'raw', 'doppler': 'Doppler only', 'all': 'raw+Doppler'}[features]
    results: dict[str, float] = {}

    X_sim_fl  = _sklearn_featurize(X_sim)
    X_real_fl = _sklearn_featurize(X_real)
    y_sim_enc  = label_enc_sim.transform(y_sim)
    y_real_enc = label_enc_sim.transform(y_real)

    def _build():
        if arch == 'RF':
            return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        return SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

    # Scenario 1: train su sim, test su sim
    _print_section(f'{arch} — Scenario 1: train su sim / test su sim')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sim_fl, y_sim_enc, test_size=0.3, random_state=42,
    )
    print(f'  Train: {X_tr.shape}  |  Test: {X_te.shape}')
    clf = _build()
    clf.fit(X_tr, y_tr)
    y_pred_te          = clf.predict(X_te)
    results['sim_test'] = float(np.mean(y_pred_te == y_te))
    print(f'  Sim test acc: {results["sim_test"]*100:.1f}%')
    joblib.dump(clf, os.path.join(models_dir, f'{tag}_sim.pkl'))
    _plot_confusion_matrix(
        y_te, y_pred_te, label_enc_sim,
        title=f'{arch} [{feat_label}] — sim test',
        save_prefix=f'cm_{tag}_train_sim_test_sim',
        out_dir=figs_dir, accuracy=results['sim_test'],
    )

    # Scenario 2: zero-shot su reale (stesso modello addestrato su sim)
    _print_section(f'{arch} — Scenario 2: zero-shot su reale')
    print(f'  Real: {X_real_fl.shape}')
    y_pred_real            = clf.predict(X_real_fl)
    results['zeroshot_real'] = float(np.mean(y_pred_real == y_real_enc))
    print(f'  Zero-shot acc: {results["zeroshot_real"]*100:.1f}%')
    _plot_confusion_matrix(
        y_real_enc, y_pred_real, label_enc_sim,
        title=f'{arch} [{feat_label}] — zero-shot real',
        save_prefix=f'cm_{tag}_train_sim_test_real',
        out_dir=figs_dir, accuracy=results['zeroshot_real'],
    )

    # Scenario 3: train su reale, test su reale
    if enable_real_training:
        _print_section(f'{arch} — Scenario 3: train su reale / test su reale')
        le_real     = LabelEncoder().fit(y_real)
        y_real_enc2 = le_real.transform(y_real)
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            X_real_fl, y_real_enc2, test_size=0.3, random_state=42,
        )
        print(f'  Train: {X_tr_r.shape}  |  Test: {X_te_r.shape}')
        clf_r = _build()
        clf_r.fit(X_tr_r, y_tr_r)
        y_pred_r            = clf_r.predict(X_te_r)
        results['real_trained'] = float(np.mean(y_pred_r == y_te_r))
        print(f'  Real trained acc: {results["real_trained"]*100:.1f}%')
        _plot_confusion_matrix(
            y_te_r, y_pred_r, le_real,
            title=f'{arch} [{feat_label}] — trained on real',
            save_prefix=f'cm_{tag}_train_real_test_real',
            out_dir=figs_dir, accuracy=results['real_trained'],
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def setup_ddp(gpu: str | None) -> tuple:
    """Inizializza DDP se lanciato con torchrun, altrimenti single-GPU/CPU.

    Returns: (device, rank, world_size)
      - Con torchrun: LOCAL_RANK determina la GPU di questo processo.
      - Senza torchrun: rank=0, world_size=1, comportamento identico al single-GPU.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    if local_rank >= 0:
        # Modalità DDP — lanciato con torchrun
        if gpu is not None:
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', device_id=device,
                                timeout=datetime.timedelta(hours=4))
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        # cuDNN check su ogni rank (stesso problema del path single-GPU)
        try:
            nn.LSTM(1, 1).to(device)
        except RuntimeError as exc:
            if 'cuDNN' in str(exc):
                torch.backends.cudnn.enabled = False
                if rank == 0:
                    print('[WARN] cuDNN incompatibile — disabilitato')
        if rank == 0:
            names = [torch.cuda.get_device_name(i) for i in range(world_size)]
            print(f'DDP attivo: {world_size} GPU  {names}')
    else:
        # Modalità single-GPU / CPU
        if gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if torch.cuda.is_available():
            device = torch.device('cuda')
            names  = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f'GPU disponibili: {names}  — uso cuda:0')
            try:
                nn.LSTM(1, 1).to(device)
            except RuntimeError as exc:
                if 'cuDNN' in str(exc):
                    torch.backends.cudnn.enabled = False
                    print('[WARN] cuDNN incompatibile — disabilitato')
        else:
            device = torch.device('cpu')
            print('Nessuna GPU trovata — uso CPU.')
        rank = 0
        world_size = 1

    return device, rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--arch',
        default='both',
        help='Architettura/e: cnn, lstm, transformer, rf, svm — separati da virgola; '
             'oppure both (=cnn,lstm) o all (=tutti)',
    )
    parser.add_argument(
        '--activities',
        default=','.join(DEFAULT_ACTIVITIES),
        help='Classi da includere, separate da virgola',
    )
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Salta il training: genera solo le confusion matrix dai modelli .pt già presenti',
    )
    parser.add_argument(
        '--train-real', action='store_true',
        help='Abilita anche lo Scenario 3: train su reale / test su reale (default: disabilitato)',
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Rigenera dataset_har.csv dai file raw anche se già presente',
    )
    parser.add_argument(
        '--features', default='all',
        help='raw,doppler,all separati da virgola. Es. --features doppler,all  (default: all)',
    )
    parser.add_argument(
        '--stride-sc', type=int, default=1, metavar='S',
        help='Sottocampiona subcarrier ogni S passi (es. 4 → 1272→318 SC). Default: 1 (nessun subsampling)',
    )
    parser.add_argument(
        '--gpu', default=None,
        help='GPU da usare, es. 0, 1, 0,1  (default: tutte le GPU disponibili)',
    )
    parser.add_argument(
        '--dataset-sim', default=DIR_DATASET_SIM,
        help=f'Directory dataset simulato (default: {DIR_DATASET_SIM})',
    )
    parser.add_argument(
        '--dataset-real', default=DIR_DATASET_REAL,
        help=f'Directory dataset reale (default: {DIR_DATASET_REAL})',
    )
    parser.add_argument(
        '--rx-sim', type=int, default=-1, metavar='N',
        help='Receiver SIM da usare (0-3); -1 = tutti i receiver (default: -1)',
    )
    parser.add_argument(
        '--subtract-mode',
        choices=['none', 'dt', 'dt_ema', 'combined'],
        default='none',
        help='Modalità sottrazione H_s: none=H_activity (default), '
             'dt=DT prior puro, dt_ema=EMA inizializzata col prior DT, '
             'combined=H_combined grezzo (H_s+H_d) senza sottrazione, le '
             'feature Doppler cancellano H_s per costruzione',
    )
    parser.add_argument(
        '--ema-alpha', type=float, default=EMA_ALPHA, metavar='A',
        help=f'Fattore smoothing EMA per --subtract-mode dt_ema (default: {EMA_ALPHA})',
    )
    args = parser.parse_args()
    subtract_mode = args.subtract_mode
    global APPLY_STATIC_SUBTRACTION
    APPLY_STATIC_SUBTRACTION = (subtract_mode != 'none')
    rx_sim_filter = None if args.rx_sim < 0 else [args.rx_sim]

    activities  = [a.strip().upper() for a in args.activities.split(',') if a.strip()]
    _VALID_FEAT = {'raw', 'doppler', 'all'}
    features_list = [f.strip() for f in args.features.split(',') if f.strip()]
    invalid = [f for f in features_list if f not in _VALID_FEAT]
    if invalid:
        parser.error(f'--features valori non validi: {invalid}  (scegli tra: raw, doppler, all)')
    _DEEP_ARCHS    = ['CNN', 'LSTM', 'TRANSFORMER']
    _SKLEARN_ARCHS = ['RF', 'SVM']
    _ALL_ARCHS     = _DEEP_ARCHS + _SKLEARN_ARCHS
    _VALID_ARCHS   = {a.lower() for a in _ALL_ARCHS} | {'both', 'all'}
    if args.arch == 'both':
        archs = ['CNN', 'LSTM']
    elif args.arch == 'all':
        archs = _ALL_ARCHS
    else:
        tokens = [t.strip().lower() for t in args.arch.split(',') if t.strip()]
        invalid = [t for t in tokens if t not in _VALID_ARCHS - {'both', 'all'}]
        if invalid:
            parser.error(f'--arch valori non validi: {invalid}  (scegli tra: {", ".join(sorted(_VALID_ARCHS))})')
        archs = [t.upper() for t in tokens]
    acts_tag    = '_'.join(sorted(activities))
    sc_tag      = f'_sc{args.stride_sc}' if args.stride_sc > 1 else ''
    sub_tag     = '' if subtract_mode == 'none' else f'_{subtract_mode}'
    dataset_tag = Path(args.dataset_sim).name + '-' + Path(args.dataset_real).name + sc_tag + sub_tag
    figs_dir   = os.path.join('figs',   dataset_tag, acts_tag)
    models_dir = os.path.join('models', dataset_tag, acts_tag)
    # Rank non ancora noto; usiamo LOCAL_RANK direttamente (0 in modalità single-GPU)
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        os.makedirs(figs_dir,   exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    device, rank, world_size = setup_ddp(args.gpu)
    if rank == 0:
        print(f'PyTorch {torch.__version__}')
        if world_size > 1:
            print(f'Batch effettivo per architettura: {args.batch_size * world_size} '
                  f'({args.batch_size} × {world_size} GPU)')

    # In DDP: rank 0 fa tutto il pre-processing pesante (CSV → .npz), barrier,
    # poi tutti i rank caricano dal .npz (fast path, quasi istantaneo).
    # Questo copre: rebuild esplicito, CSV mancante, .npz mancante.
    if world_size > 1:
        if rank == 0:
            _print_section('Pre-processing dataset simulato')
            print(f'  Base dir : {args.dataset_sim}')
            print(f'  Attività : {activities}')
            X_sim, y_sim, rx_sim = load_dataset(
                args.dataset_sim, activities,
                rx_indices=rx_sim_filter,
                rebuild=args.rebuild,
            )
            print(f'  rx_id    : {np.unique(rx_sim)}')
            _print_section('Pre-processing dataset reale')
            print(f'  Base dir : {args.dataset_real}')
            X_real, y_real, rx_real = load_dataset(
                args.dataset_real, activities,
                rx_indices=None,
                rebuild=args.rebuild,
            )
        dist.barrier()   # rank 1-3 aspettano qui finché rank 0 ha scritto i .npz
        if rank != 0:
            X_sim, y_sim, rx_sim = load_dataset(
                args.dataset_sim, activities,
                rx_indices=rx_sim_filter,
                rebuild=False,
            )
            X_real, y_real, rx_real = load_dataset(
                args.dataset_real, activities,
                rx_indices=None,
                rebuild=False,
            )
    else:
        # Single-GPU: carica normalmente
        if rank == 0:
            _print_section('Pre-processing dataset simulato')
            print(f'  Base dir : {args.dataset_sim}')
            print(f'  Attività : {activities}')
        X_sim, y_sim, rx_sim = load_dataset(
            args.dataset_sim, activities,
            rx_indices=rx_sim_filter,
            rebuild=args.rebuild,
        )
        if rank == 0:
            print(f'  rx_id    : {np.unique(rx_sim)}')
            _print_section('Pre-processing dataset reale')
            print(f'  Base dir : {args.dataset_real}')
        X_real, y_real, rx_real = load_dataset(
            args.dataset_real, activities,
            rx_indices=None,
            rebuild=args.rebuild,
        )

    # ── Rimozione componente statica ──────────────────────────────────────────
    if subtract_mode == 'dt':
        if rank == 0:
            _print_section('Rimozione H_s — DT prior puro (zero-shot)')
            print('  SIM  → H_s^rx stimato per-receiver dai campioni IDLE simulati')
            print('  REAL → zero-shot: H_s^sim applicato al REAL')
        X_sim, H_s_map_sim = subtract_static_prior(X_sim, y_sim, rx_sim)
        X_real, _          = subtract_static_prior(X_real, y_real, rx_real,
                                                    H_s_map=H_s_map_sim)
    elif subtract_mode == 'dt_ema':
        if rank == 0:
            _print_section('Rimozione H_s — DT-initialized EMA')
            print(f'  alpha={args.ema_alpha}  |  '
                  f'H_bg[0]=H_s^DT, H_bg[t]=alpha*H_bg[t-1]+(1-alpha)*H[t]')
            print('  SIM  → EMA warm-started da H_s^DT^sim')
            print('  REAL → EMA warm-started da H_s^DT^sim (zero-shot, '
                  'si adatta al canale reale nel tempo)')
        X_sim, H_s_map_sim = apply_dt_ema(X_sim, y_sim, rx_sim, alpha=args.ema_alpha)
        X_real, _          = apply_dt_ema(X_real, y_real, rx_real,
                                           H_s_map=H_s_map_sim, alpha=args.ema_alpha)
    elif subtract_mode == 'combined':
        if rank == 0:
            _print_section('Dataset H_combined grezzo (H_s+H_d, nessuna sottrazione)')
            print('  SIM/REAL → H_combined_v*.csv usato as-is: le feature Doppler '
                  '(differenze temporali) cancellano H_s per costruzione, senza '
                  'bisogno di stimarlo via DT/EMA')
    else:  # none
        if rank == 0:
            _print_section('Dataset H_activity (H_d già pulita — nessuna sottrazione aggiuntiva)')
            print('  SIM  → H_activity_v*.csv: H_s già rimossa al momento della sim')

    label_enc_sim = LabelEncoder().fit(y_sim)

    # Dati raw conservati per poter applicare feature set diversi in sequenza
    X_sim_raw  = X_sim
    X_real_raw = X_real

    # ── Loop su feature set richiesti ─────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for features in features_list:
        if rank == 0:
            _print_section(f'FEATURE SET: {features}')

        X_sim_f  = X_sim_raw.copy()
        X_real_f = X_real_raw.copy() if X_real_raw is not None else None

        # Feature engineering Doppler
        if features in ('doppler', 'all'):
            if rank == 0:
                print('  Calcolo features Doppler scale-invarianti...')
            X_sim_f = augment_features(X_sim_f, demean_amp=(subtract_mode == 'combined'))
            if X_real_f is not None:
                X_real_f = augment_features(X_real_f, demean_amp=(subtract_mode == 'combined'))
            if features == 'doppler':
                X_sim_f = X_sim_f[..., 4:]
                if X_real_f is not None:
                    X_real_f = X_real_f[..., 4:]
        if rank == 0:
            print(f'  Shape sim : {X_sim_f.shape}')

        # Subcarrier subsampling
        if args.stride_sc > 1:
            X_sim_f = X_sim_f[:, :, ::args.stride_sc, :]
            if X_real_f is not None:
                X_real_f = X_real_f[:, :, ::args.stride_sc, :]
            if rank == 0:
                print(f'  Shape sim dopo subsampling: {X_sim_f.shape}')

        # Normalizzazione per-feature (fit su sim, apply su sim e real)
        orig_shape = X_sim_f.shape
        n_feat     = orig_shape[-1]
        scaler     = StandardScaler()
        X_sim_f    = scaler.fit_transform(
            X_sim_f.reshape(-1, n_feat)
        ).reshape(orig_shape).astype(np.float32)
        if X_real_f is not None:
            real_shape = X_real_f.shape
            X_real_f   = scaler.transform(
                X_real_f.reshape(-1, n_feat)
            ).reshape(real_shape).astype(np.float32)
        if rank == 0:
            print(f'  Scaler fit su sim ({orig_shape}) → applicato a sim e real')

        # Benchmark per ogni architettura
        feat_results: dict[str, dict] = {}
        for arch in archs:
            if rank == 0:
                _print_section(f'ARCHITETTURA: {arch}  [{features}]')
            if arch in _SKLEARN_ARCHS:
                feat_results[arch] = run_sklearn_arch(
                    arch, X_sim_f, y_sim, X_real_f, y_real,
                    label_enc_sim, figs_dir, models_dir,
                    features=features,
                    enable_real_training=args.train_real,
                    rank=rank,
                    subtract_tag=sub_tag.lstrip('_'),
                )
            else:
                feat_results[arch] = run_architecture(
                    arch, X_sim_f, y_sim, X_real_f, y_real,
                    label_enc_sim, figs_dir, models_dir, args.epochs, args.batch_size,
                    features=features,
                    eval_only=args.eval_only,
                    enable_real_training=args.train_real,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    subtract_tag=sub_tag.lstrip('_'),
                )
        all_results[features] = feat_results

    # ── Riepilogo (solo rank 0) ───────────────────────────────────────────────
    if rank == 0:
        _print_section('RIEPILOGO ACCURATEZZE')
        for features, feat_res in all_results.items():
            print(f'\n  Features: {features}')
            header = f'  {"Architettura":<10}  {"Sim test":>10}  {"Zero-shot":>10}  {"Reale trained":>14}'
            print(header)
            print('  ' + '-' * (len(header) - 2))
            for arch, res in feat_res.items():
                sim = f'{res.get("sim_test",      float("nan")) * 100:.1f}%'
                zs  = f'{res.get("zeroshot_real", float("nan")) * 100:.1f}%'
                rt  = f'{res.get("real_trained",  float("nan")) * 100:.1f}%'
                print(f'  {arch:<10}  {sim:>10}  {zs:>10}  {rt:>14}')
        print(f'\nFigure  → {figs_dir}/')
        print(f'Modelli → {models_dir}/')

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
