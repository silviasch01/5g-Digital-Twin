#!/usr/bin/env python3
"""
csi_har_benchmark.py
Benchmark CNN e Bidirectional-LSTM per Human Activity Recognition (HAR)
su dati CSI (Channel State Information) simulati e reali.

Pipeline dataset (equivalente a dataset.ipynb + notebook HAR):
  1. Legge H_activity_v*.csv raw da dataset_v2/ / dataset_real_v2/
  2. Costruisce e salva dataset_har.csv (cached; rigenerato con --rebuild)
  3. Carica da dataset_har.csv per il training (stesso flusso dei notebook)

Scenari eseguiti per ogni architettura:
  1. Train su sim (70/30) → test su sim
  2. Zero-shot: modello addestrato su sim applicato al dataset reale
  3. Train su reale (70/30) → test su reale

Uso:
    python csi_har_benchmark.py                          # CNN + LSTM, 4 classi default
    python csi_har_benchmark.py --arch cnn               # solo CNN
    python csi_har_benchmark.py --epochs 5 --arch lstm   # test rapido LSTM
    python csi_har_benchmark.py --activities FALL,RUN,STAND,WALK
    python csi_har_benchmark.py --rebuild                # rigenera dataset_har.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Sopprime i log C++ di TensorFlow/abseil prima di qualsiasi import TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('ABSL_LOGTOSTDERR', '0')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE — modifica qui per cambiare dataset o parametri di default
# ═══════════════════════════════════════════════════════════════════════════════

# ── Percorsi dataset (relativi alla directory dello script) ──────────────────
DIR_DATASET_SIM  = "dataset_v2"        # simulato: 4 posizioni RX
DIR_DATASET_REAL = "dataset_real_v2"   # reale:    posizione RX fissa

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
N_FEATURES_DOPPLER = 8     # raw + 4 Doppler: dPHA, |dPHA|, dH_abs, PHA_slope
N_RX_SIM           = 4     # posizioni RX simulate (rx0 … rx3)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING E PRE-PROCESSING  (equivalente a dataset.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Serializzazione / deserializzazione (formato dataset_har.csv) ────────────

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


def _array_to_dataset_string(H: np.ndarray) -> str:
    """(N_STEPS, N_SC, N_FEATURES) → stringa '[\nstep0\n...\nstepN\n]'.

    Formato identico a quello usato da dataset.ipynb e dai notebook HAR.
    """
    def step_row(h):
        return ';'.join(','.join(f'{v:.6f}' for v in h[i]) for i in range(len(h)))
    body = '\n'.join(step_row(H[s]) for s in range(len(H)))
    return f'[\n{body}\n]'


def _dataset_string_to_array(s: str) -> np.ndarray:
    """'[\nstep0\n...\nstepN\n]' → (N_STEPS, N_SC, N_FEATURES) float32."""
    inner = s.strip()[1:-1].strip()
    lines = inner.split('\n')
    return np.array(
        [[list(map(float, sc.split(','))) for sc in line.split(';')] for line in lines],
        dtype=np.float32,
    )


# ── Costruzione dataset_har.csv (dataset.ipynb step 1) ──────────────────────

def build_dataset_csv(
    base_dir: str,
    activities: list[str],
    rx_indices: list[int] | None,
    out_path: str,
) -> None:
    """Legge H_activity_v*.csv raw e scrive dataset_har.csv.

    Equivalente alla cella principale di dataset.ipynb.
    Fix rispetto al notebook: usa ACTIVITY_DIRS (path corretti, IDLE incluso).

    Args:
        base_dir:   root dataset (es. "dataset_v2").
        activities: classi da includere.
        rx_indices: [0,1,2,3] per sim, None per reale (posizione fissa).
        out_path:   percorso del CSV di output.
    """
    base = Path(base_dir)
    records = []
    sample_id = 0

    for act in activities:
        act_dir = base / ACTIVITY_DIRS[act]

        if rx_indices is not None:
            search_dirs = [(act_dir / f'rx{rx}', rx) for rx in rx_indices]
        else:
            search_dirs = [(act_dir, 0)]

        n_act = 0
        for dir_path, rx_id in search_dirs:
            raw_files = sorted(dir_path.glob('H_activity_v*.csv'))
            if not raw_files:
                print(f'  [WARN] nessun file in {dir_path}', file=sys.stderr)
                continue
            for raw_path in raw_files:
                H = _raw_csv_to_array(str(raw_path))
                if H.shape != (N_STEPS, N_SC, N_FEATURES_RAW):
                    print(f'  [WARN] shape inattesa {H.shape} — {raw_path}', file=sys.stderr)
                    continue
                records.append({
                    'sample_id': sample_id,
                    'label':     act,
                    'rx_id':     rx_id,
                    'data':      _array_to_dataset_string(H),
                })
                sample_id += 1
                n_act += 1

        print(f'  {act}: {n_act} sample')

    df = pd.DataFrame(records).set_index(['sample_id', 'label', 'rx_id'])
    df.to_csv(out_path, sep=';')
    print(f'  → {out_path}  ({sample_id} sample totali)')


# ── Caricamento da dataset_har.csv (dataset.ipynb step 2 + notebook HAR) ────

def load_dataset(
    base_dir: str,
    activities: list[str],
    rx_indices: list[int] | None = None,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carica dataset da dataset_har.csv, costruendolo dai raw se assente.

    Flusso identico a dataset.ipynb seguito dai notebook LSTM/CNN:
      raw H_activity_v*.csv → dataset_har.csv → numpy array

    Args:
        base_dir:   root dataset (es. "dataset_v2").
        activities: classi da includere.
        rx_indices: [0,1,2,3] per sim, None per reale.
        rebuild:    forza la rigenerazione di dataset_har.csv.

    Returns:
        X      : (N, N_STEPS, N_SC, N_FEATURES) float32
        y      : (N,) array di stringhe (etichette)
        rx_ids : (N,) int
    """
    csv_path = Path(base_dir) / 'dataset_har.csv'

    if not rebuild and csv_path.exists():
        # controlla che tutte le attività richieste siano presenti nella cache
        df_check = pd.read_csv(str(csv_path), sep=';', index_col=['sample_id', 'label', 'rx_id'])
        cached_labels = set(df_check.index.get_level_values('label').unique())
        missing = set(activities) - cached_labels
        if missing:
            print(f'  [INFO] attività mancanti nella cache ({missing}) — ricostruzione...')
            rebuild = True

    if rebuild or not csv_path.exists():
        print(f'  Costruzione {csv_path} dai file raw...')
        build_dataset_csv(base_dir, activities, rx_indices, str(csv_path))
        df_check = pd.read_csv(str(csv_path), sep=';', index_col=['sample_id', 'label', 'rx_id'])
    else:
        print(f'  Caricamento da cache: {csv_path}')

    df = df_check
    # filtra le sole attività richieste (il CSV potrebbe contenerne di più)
    mask = df.index.get_level_values('label').isin(activities)
    df = df[mask]

    X_list, y_list, rx_list = [], [], []
    for (sid, label, rx_id), row in df.iterrows():
        X_list.append(_dataset_string_to_array(row['data']))
        y_list.append(label)
        rx_list.append(rx_id)

    X      = np.stack(X_list)
    y      = np.array(y_list)
    rx_ids = np.array(rx_list, dtype=np.int32)

    print(f'  X shape : {X.shape}')
    print(f'  Classi  : {np.unique(y)}')
    return X, y, rx_ids


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING DOPPLER  (da doppler_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def augment_features(X: np.ndarray) -> np.ndarray:
    """Arricchisce l'input con feature Doppler derivate dalla fase del canale.

    Input : (N, N_STEPS, N_SC, 4)  — canali raw: RE, IM, AMP, PHA
    Output: (N, N_STEPS, N_SC, 8)  — raw + 4 canali Doppler:
        [4] dPHA      — derivata temporale fase unwrapped (velocità Doppler istantanea)
        [5] abs_dPHA  — |dPHA|  (intensità movimento, distingue statico vs dinamico)
        [6] dH_abs    — |ΔH_complesso| (variazione totale del canale)
        [7] PHA_slope — pendenza lineare φ(t) per subcarrier (velocità Doppler netta
                        con segno: utile per distinguere FALL da JUMP e WALK da IDLE)
    """
    N, T, S, _ = X.shape

    H_cx   = X[..., 0].astype(np.float32) + 1j * X[..., 1].astype(np.float32)
    PHA_uw = np.unwrap(X[..., 3].astype(np.float32), axis=1)   # (N, T, S)

    # dPHA: variazione di fase tra step consecutivi — pad con 0 al t=0
    _dPHA  = np.diff(PHA_uw, axis=1)                            # (N, T-1, S)
    dPHA   = np.concatenate(
        [np.zeros((N, 1, S), np.float32), _dPHA], axis=1,
    )                                                            # (N, T, S)

    abs_dPHA = np.abs(dPHA)

    # dH_abs: modulo della variazione del canale complesso — pad con 0 al t=0
    _dH    = np.abs(np.diff(H_cx, axis=1)).astype(np.float32)   # (N, T-1, S)
    dH_abs = np.concatenate(
        [np.zeros((N, 1, S), np.float32), _dH], axis=1,
    )                                                            # (N, T, S)

    # PHA_slope: pendenza lineare di φ(t) per (sample, subcarrier)
    # Regressione lineare analitica: β = Σ (t-t̄)(φ-φ̄) / Σ (t-t̄)²
    t   = np.arange(T, dtype=np.float32)
    t_c = t - t.mean()                                          # (T,)
    denom = float((t_c ** 2).sum())
    phi_c = PHA_uw - PHA_uw.mean(axis=1, keepdims=True)        # (N, T, S)
    slope = (phi_c * t_c[None, :, None]).sum(axis=1) / denom   # (N, S)
    PHA_slope = np.broadcast_to(
        slope[:, None, :], (N, T, S),
    ).copy().astype(np.float32)                                 # (N, T, S)

    return np.concatenate([
        X.astype(np.float32),
        dPHA[..., None],
        abs_dPHA[..., None],
        dH_abs[..., None],
        PHA_slope[..., None],
    ], axis=-1)   # (N, T, S, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITETTURE
# ═══════════════════════════════════════════════════════════════════════════════

def build_cnn(input_shape: tuple, num_classes: int, strategy=None):
    """CNN 2D per HAR su CSI — input_shape=(N_STEPS, N_SC, N_FEATURES).

    strategy: tf.distribute.Strategy (opzionale). Se fornito, il modello viene
    costruito dentro strategy.scope() per abilitare il training multi-GPU.
    """
    import contextlib
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
    )
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.regularizers import l2

    ctx = strategy.scope() if strategy is not None else contextlib.nullcontext()
    with ctx:
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(64,  (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(synchronized=True),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(synchronized=True),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(synchronized=True),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(synchronized=True),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.01), dilation_rate=(2, 2)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(synchronized=True),
            MaxPooling2D((2, 2), padding='same'),

            Flatten(),
            Dense(512, kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            Dropout(0.4),

            Dense(128, kernel_regularizer=l2(0.01)),
            LeakyReLU(negative_slope=0.1),
            Dropout(0.4),

            Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=AdamW(learning_rate=5e-5, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
    return model


def build_lstm(input_shape: tuple, num_classes: int, strategy=None):
    """Bidirectional LSTM per HAR su CSI — input_shape=(N_STEPS, N_SC*N_FEATURES).

    strategy: tf.distribute.Strategy (opzionale). Se fornito, il modello viene
    costruito dentro strategy.scope() per abilitare il training multi-GPU.
    """
    import contextlib
    from tensorflow.keras.layers import (
        BatchNormalization, Bidirectional, Dense, Dropout, Input, LSTM, TimeDistributed,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import AdamW

    ctx = strategy.scope() if strategy is not None else contextlib.nullcontext()
    with ctx:
        model = Sequential([
            Input(shape=input_shape),
            TimeDistributed(Dense(128, activation='relu')),

            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(synchronized=True),
            Dropout(0.3),

            Bidirectional(LSTM(64, return_sequences=False)),
            BatchNormalization(synchronized=True),
            Dropout(0.3),

            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: str,
    epochs: int = 200,
    batch_size: int = 32,
):
    """Addestra model e salva il best checkpoint in save_path. Ritorna history.

    Usa tf.data.Dataset con drop_remainder=True per garantire batch uniformi su
    tutte le GPU (necessario con MirroredStrategy per evitare shape mismatch).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    # drop_remainder=True scarta l'ultimo batch parziale → tutti i batch hanno
    # esattamente batch_size campioni, divisibili equamente tra le GPU.
    ds_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=42)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy', mode='max',
                        save_best_only=True, verbose=1),
    ]
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# VALUTAZIONE E PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(
    model_path: str,
    X: np.ndarray,
    y_enc: np.ndarray,
    label_enc: LabelEncoder,
    title: str,
    save_prefix: str,
    out_dir: str,
) -> float:
    """Carica il best model, predice su X, plotta e salva la confusion matrix.

    Returns:
        accuracy (float, 0–1)
    """
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    accuracy = np.mean(y_pred == y_enc)
    print(f'  Accuratezza [{title}]: {accuracy * 100:.1f}%')

    cm = confusion_matrix(y_enc, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    class_labels = label_enc.classes_
    fig, ax = plt.subplots(figsize=(max(6, len(class_labels)), max(5, len(class_labels) - 1)))
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={'size': 12},
        ax=ax,
    )
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_title(f'{title}  acc={accuracy * 100:.1f}%', fontsize=13)
    ax.tick_params(axis='x', labelsize=11, rotation=30)
    ax.tick_params(axis='y', labelsize=11, rotation=0)
    fig.tight_layout()

    pdf_path = os.path.join(out_dir, f'{save_prefix}.pdf')
    png_path = os.path.join(out_dir, f'{save_prefix}.png')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Confusion matrix salvata → {png_path}')

    return accuracy


def _print_section(title: str) -> None:
    print(f'\n{"═" * 70}')
    print(f'  {title}')
    print(f'{"═" * 70}')


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
    out_dir: str,
    epochs: int,
    batch_size: int,
    features: str = 'raw',
    strategy=None,
) -> dict[str, float]:
    """Esegue i 3 scenari per un'architettura (CNN o LSTM).

    Returns:
        dict con chiavi 'sim_test', 'zeroshot_real', 'real_trained'
    """
    from tensorflow.keras.utils import to_categorical

    tag     = f'{arch.lower()}_{features}'   # es. cnn_doppler, lstm_raw
    feat_label = 'Doppler' if features == 'doppler' else 'raw'
    results = {}

    # ── Input shape — ricavato da X_sim (funziona sia con raw che con Doppler)
    n_feat = X_sim.shape[-1]   # 4 (raw) o 8 (doppler)

    def maybe_reshape(X):
        if arch == 'LSTM':
            return X.reshape(len(X), N_STEPS, N_SC * n_feat)
        return X

    if arch == 'CNN':
        input_shape = (N_STEPS, N_SC, n_feat)
    else:
        input_shape = (N_STEPS, N_SC * n_feat)

    num_classes = len(label_enc_sim.classes_)

    # ────────────────────────────────────────────────────────────────────────
    # Scenario 1 & 2: train su sim → test su sim, poi zero-shot su reale
    # ────────────────────────────────────────────────────────────────────────
    _print_section(f'{arch} — Scenario 1: train su sim / test su sim')

    y_sim_enc = label_enc_sim.transform(y_sim)
    y_sim_cat = to_categorical(y_sim_enc, num_classes)

    idx = np.arange(len(X_sim))
    np.random.seed(42)
    np.random.shuffle(idx)
    X_sim_sh = X_sim[idx]
    y_sim_sh  = y_sim_cat[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sim_sh, y_sim_sh, test_size=0.3, random_state=42,
    )
    X_tr_r, X_te_r = maybe_reshape(X_tr), maybe_reshape(X_te)

    print(f'\n  Train: {X_tr_r.shape}  |  Test: {X_te_r.shape}')
    _print_class_dist('  Train', label_enc_sim, np.argmax(y_tr, axis=1))
    _print_class_dist('  Test ', label_enc_sim, np.argmax(y_te, axis=1))

    model_sim_path = os.path.join(out_dir, f'{tag}_sim.h5')
    model = build_cnn(input_shape, num_classes, strategy) if arch == 'CNN' \
            else build_lstm(input_shape, num_classes, strategy)
    model.summary()
    train_and_save(model, X_tr_r, y_tr, X_te_r, y_te, model_sim_path, epochs, batch_size)

    results['sim_test'] = evaluate_and_plot(
        model_sim_path, X_te_r, np.argmax(y_te, axis=1),
        label_enc_sim,
        title=f'{arch} [{feat_label}] — sim test',
        save_prefix=f'cm_{tag}_sim_test',
        out_dir=out_dir,
    )

    # ── Scenario 2: zero-shot su reale ──────────────────────────────────────
    _print_section(f'{arch} — Scenario 2: zero-shot su dataset reale')

    y_real_enc_sim = label_enc_sim.transform(y_real)
    X_real_r = maybe_reshape(X_real)

    results['zeroshot_real'] = evaluate_and_plot(
        model_sim_path, X_real_r, y_real_enc_sim,
        label_enc_sim,
        title=f'{arch} [{feat_label}] — zero-shot reale',
        save_prefix=f'cm_{tag}_zeroshot_real',
        out_dir=out_dir,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Scenario 3: train su reale → test su reale
    # ────────────────────────────────────────────────────────────────────────
    _print_section(f'{arch} — Scenario 3: train su reale / test su reale')

    le_real = LabelEncoder().fit(y_real)
    y_real_enc = le_real.transform(y_real)
    y_real_cat = to_categorical(y_real_enc, len(le_real.classes_))

    idx_r = np.arange(len(X_real))
    np.random.seed(42)
    np.random.shuffle(idx_r)
    X_real_sh = X_real[idx_r]
    y_real_sh  = y_real_cat[idx_r]

    X_tr_r2, X_te_r2, y_tr_r2, y_te_r2 = train_test_split(
        X_real_sh, y_real_sh, test_size=0.3, random_state=42,
    )
    X_tr_r2, X_te_r2 = maybe_reshape(X_tr_r2), maybe_reshape(X_te_r2)

    print(f'\n  Train: {X_tr_r2.shape}  |  Test: {X_te_r2.shape}')
    _print_class_dist('  Train', le_real, np.argmax(y_tr_r2, axis=1))
    _print_class_dist('  Test ', le_real, np.argmax(y_te_r2, axis=1))

    model_real_path = os.path.join(out_dir, f'{tag}_real.h5')
    model_r = build_cnn(input_shape, len(le_real.classes_), strategy) if arch == 'CNN' \
              else build_lstm(input_shape, len(le_real.classes_), strategy)
    train_and_save(model_r, X_tr_r2, y_tr_r2, X_te_r2, y_te_r2,
                   model_real_path, epochs, batch_size)

    results['real_trained'] = evaluate_and_plot(
        model_real_path, X_te_r2, np.argmax(y_te_r2, axis=1),
        le_real,
        title=f'{arch} [{feat_label}] — addestrato su reale',
        save_prefix=f'cm_{tag}_real_trained',
        out_dir=out_dir,
    )

    return results


def _print_class_dist(prefix: str, le: LabelEncoder, y_enc: np.ndarray) -> None:
    classes, counts = np.unique(y_enc, return_counts=True)
    dist = '  '.join(f'{le.classes_[c]}:{n}' for c, n in zip(classes, counts))
    print(f'{prefix}  {dist}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def setup_gpu():
    """Configura le GPU e restituisce una tf.distribute.Strategy.

    - 0 GPU  → OneDeviceStrategy('/cpu:0')
    - 1 GPU  → OneDeviceStrategy('/gpu:0')   (stessa semantica di prima)
    - N GPU  → MirroredStrategy su tutte le GPU (data-parallel)
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'MirroredStrategy su {strategy.num_replicas_in_sync} GPU: {[g.name for g in gpus]}')
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy('/gpu:0')
        print(f'GPU singola: {gpus[0].name}')
    else:
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
        print('Nessuna GPU trovata — uso CPU.')
    return strategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--arch', choices=['cnn', 'lstm', 'both'], default='both',
        help='Architettura da addestrare (default: both)',
    )
    parser.add_argument(
        '--activities',
        default=','.join(DEFAULT_ACTIVITIES),
        help='Classi da includere, separate da virgola '
             '(default: FALL,RUN,STAND,WALK — come il notebook LSTM su dataset_v2)',
    )
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--out-dir', default='outputs_har',
        help='Directory di output per modelli e confusion matrix',
    )
    parser.add_argument(
        '--skip-real', action='store_true',
        help='Salta il caricamento e i 2 scenari sul dataset reale',
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Rigenera dataset_har.csv dai file raw anche se già presente',
    )
    parser.add_argument(
        '--features', choices=['raw', 'doppler'], default='doppler',
        help='raw: (N,40,1272,4)  doppler: aggiunge dPHA,|dPHA|,dH_abs,PHA_slope → (N,40,1272,8)',
    )
    args = parser.parse_args()

    activities = [a.strip().upper() for a in args.activities.split(',') if a.strip()]
    archs = ['CNN', 'LSTM'] if args.arch == 'both' else [args.arch.upper()]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Setup TF / GPU ───────────────────────────────────────────────────────
    import tensorflow as tf
    print(f'TensorFlow {tf.__version__}')
    strategy    = setup_gpu()
    n_replicas  = strategy.num_replicas_in_sync
    effective_batch = args.batch_size * n_replicas
    if n_replicas > 1:
        print(f'  Batch size per replica: {args.batch_size}  ×  {n_replicas} GPU  →  effettivo: {effective_batch}')

    # ── Pre-processing + caricamento dataset simulato ────────────────────────
    _print_section('Pre-processing dataset simulato')
    print(f'  Base dir : {DIR_DATASET_SIM}')
    print(f'  Attività : {activities}')
    X_sim, y_sim, rx_sim = load_dataset(
        DIR_DATASET_SIM, activities,
        rx_indices=list(range(N_RX_SIM)),
        rebuild=args.rebuild,
    )
    print(f'  rx_id    : {np.unique(rx_sim)}')

    label_enc_sim = LabelEncoder().fit(y_sim)

    # ── Pre-processing + caricamento dataset reale ───────────────────────────
    if not args.skip_real:
        _print_section('Pre-processing dataset reale')
        print(f'  Base dir : {DIR_DATASET_REAL}')
        X_real, y_real, _ = load_dataset(
            DIR_DATASET_REAL, activities,
            rx_indices=None,
            rebuild=args.rebuild,
        )
    else:
        X_real = y_real = None

    # ── Feature engineering Doppler (opzionale) ──────────────────────────────
    if args.features == 'doppler':
        _print_section('Feature engineering Doppler')
        print('  Calcolo dPHA, |dPHA|, dH_abs, PHA_slope...')
        X_sim = augment_features(X_sim)
        if X_real is not None:
            X_real = augment_features(X_real)
        print(f'  Nuovo shape sim : {X_sim.shape}')

    n_feat = X_sim.shape[-1]   # 4 o 8

    # ── Benchmark ────────────────────────────────────────────────────────────
    all_results = {}
    for arch in archs:
        _print_section(f'ARCHITETTURA: {arch}')
        if args.skip_real:
            # Esegui solo scenario 1 (sim→sim) senza dataset reale
            from tensorflow.keras.utils import to_categorical

            feat_label = 'Doppler' if args.features == 'doppler' else 'raw'
            tag = f'{arch.lower()}_{args.features}'
            input_shape = (N_STEPS, N_SC, n_feat) if arch == 'CNN' \
                          else (N_STEPS, N_SC * n_feat)
            num_classes = len(label_enc_sim.classes_)

            def _maybe_reshape(X):
                return X.reshape(len(X), N_STEPS, N_SC * n_feat) if arch == 'LSTM' else X

            y_enc = label_enc_sim.transform(y_sim)
            y_cat = to_categorical(y_enc, num_classes)
            idx = np.arange(len(X_sim))
            np.random.seed(42)
            np.random.shuffle(idx)
            X_sh, y_sh = X_sim[idx], y_cat[idx]
            X_tr, X_te, y_tr, y_te = train_test_split(X_sh, y_sh, test_size=0.3, random_state=42)
            X_tr, X_te = _maybe_reshape(X_tr), _maybe_reshape(X_te)

            model_path = os.path.join(out_dir, f'{tag}_sim.h5')
            m = build_cnn(input_shape, num_classes, strategy) if arch == 'CNN' \
                else build_lstm(input_shape, num_classes, strategy)
            m.summary()
            train_and_save(m, X_tr, y_tr, X_te, y_te, model_path, args.epochs, effective_batch)
            acc = evaluate_and_plot(
                model_path, X_te, np.argmax(y_te, axis=1), label_enc_sim,
                f'{arch} [{feat_label}] — sim test', f'cm_{tag}_sim_test', out_dir,
            )
            all_results[arch] = {'sim_test': acc}
        else:
            all_results[arch] = run_architecture(
                arch, X_sim, y_sim, X_real, y_real,
                label_enc_sim, out_dir, args.epochs, effective_batch,
                features=args.features, strategy=strategy,
            )

    # ── Riepilogo ────────────────────────────────────────────────────────────
    _print_section('RIEPILOGO ACCURATEZZE')
    header = f'{"Architettura":<10}  {"Sim test":>10}  {"Zero-shot":>10}  {"Reale trained":>14}'
    print(header)
    print('-' * len(header))
    for arch, res in all_results.items():
        sim   = f'{res.get("sim_test",    float("nan")) * 100:.1f}%'
        zs    = f'{res.get("zeroshot_real", float("nan")) * 100:.1f}%'
        rt    = f'{res.get("real_trained",  float("nan")) * 100:.1f}%'
        print(f'{arch:<10}  {sim:>10}  {zs:>10}  {rt:>14}')
    print(f'\nOutput salvati in: {out_dir}/')


if __name__ == '__main__':
    main()
