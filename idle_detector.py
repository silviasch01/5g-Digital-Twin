#!/usr/bin/env python3
"""
IDLE detector basato su metriche di ampiezza e fase del CSI.

Pipeline:
  H_raw(t) → [metriche] → [soglia adattiva] → IDLE / NO-IDLE
                                                    │
                                        IDLE  → aggiorna H_s
                                        NO-IDLE → H_d = H - H_s → HAR

Test:
  python idle_detector.py                          # usa dataset_v2 + dataset_real_v2
  python idle_detector.py --sim dataset_v3 --real dataset_real_v3
  python idle_detector.py --no-roc                 # salta ROC, solo boxplot + pipeline sim
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# ── Costanti ──────────────────────────────────────────────────────────────────
CLASSES      = ['FALL', 'IDLE', 'JUMP', 'RUN', 'STAND', 'WALK']
CLASS_COLORS = {
    'FALL': '#e74c3c', 'IDLE': '#3498db', 'JUMP': '#1abc9c',
    'RUN':  '#e67e22', 'STAND': '#2ecc71', 'WALK': '#f39c12',
}
OUT_DIR = 'figs/idle_detector'

# Indici feature nel dataset npz (N, T, N_SC, 4)
AMP_IDX   = 2   # |H|
PHASE_IDX = 3   # angle(H)


# ══════════════════════════════════════════════════════════════════════════════
# Metriche
# ══════════════════════════════════════════════════════════════════════════════

def motion_metric(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, N_SC, 4)
    Ritorna vettore scalare (N,) — valore alto = movimento, basso = IDLE.

    Metrica combinata:
      amp_var        = varianza temporale di |H| mediata sui subcarrier
      phase_diff_var = varianza delle diff di fase frame-to-frame mediata sui SC
    """
    amp   = X[..., AMP_IDX]                        # (N, T, N_SC)
    phase = X[..., PHASE_IDX]                      # (N, T, N_SC)

    amp_var        = amp.var(axis=1).mean(axis=1)                    # (N,)
    phase_diff_var = np.diff(phase, axis=1).var(axis=1).mean(axis=1) # (N,)

    return amp_var + phase_diff_var


def amp_var_only(X: np.ndarray) -> np.ndarray:
    return X[..., AMP_IDX].var(axis=1).mean(axis=1)


def phase_diff_var_only(X: np.ndarray) -> np.ndarray:
    return np.diff(X[..., PHASE_IDX], axis=1).var(axis=1).mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# IdleDetector — pipeline real-time simulata
# ══════════════════════════════════════════════════════════════════════════════

class IdleDetector:
    """
    Rilevatore IDLE online con soglia adattiva.

    Fase cold-start (primi n_init campioni): assume IDLE, stima la baseline
    della metrica di movimento e calcola la soglia = mean + k * std.
    Poi aggiorna la baseline solo durante i periodi confermati IDLE.

    H_s viene aggiornato via EMA sui frame IDLE confermati.
    """

    def __init__(self, n_init: int = 10, k_sigma: float = 3.0, ema_alpha: float = 0.95):
        self.n_init    = n_init
        self.k         = k_sigma
        self.alpha     = ema_alpha

        self._history  = []          # metriche durante cold-start
        self.threshold = None
        self._base_mean = None
        self._base_std  = None

        self.H_s       = None        # stima corrente del canale statico (N_SC,)
        self.n_idle    = 0
        self.n_active  = 0

    def _metric(self, X_window: np.ndarray) -> float:
        """X_window: (T, N_SC, 4) — singola finestra temporale."""
        amp_v  = X_window[..., AMP_IDX].var(axis=0).mean()
        ph_dv  = np.diff(X_window[..., PHASE_IDX], axis=0).var(axis=0).mean()
        return float(amp_v + ph_dv)

    def update(self, X_window: np.ndarray):
        """
        Aggiorna lo stato con una nuova finestra.
        X_window: (T, N_SC, 4)
        Returns: (is_idle: bool, H_s: np.ndarray | None)
        """
        m = self._metric(X_window)

        # Cold-start
        if len(self._history) < self.n_init:
            self._history.append(m)
            H_mean = X_window[..., AMP_IDX].mean(axis=0)   # (N_SC,)
            self.H_s = H_mean if self.H_s is None else (
                self.alpha * self.H_s + (1 - self.alpha) * H_mean
            )
            if len(self._history) == self.n_init:
                self._base_mean = float(np.mean(self._history))
                self._base_std  = float(np.std(self._history)) + 1e-12
                self.threshold  = self._base_mean + self.k * self._base_std
            return True, self.H_s

        is_idle = m < self.threshold

        if is_idle:
            self.n_idle += 1
            H_mean   = X_window[..., AMP_IDX].mean(axis=0)
            self.H_s = self.alpha * self.H_s + (1 - self.alpha) * H_mean
            self._base_mean = self.alpha * self._base_mean + (1 - self.alpha) * m
            self.threshold  = self._base_mean + self.k * self._base_std
        else:
            self.n_active += 1

        return is_idle, self.H_s


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(npz_path: str):
    print(f'Loading {npz_path} ...')
    d = np.load(npz_path)
    X, y = d['X'].astype(np.float32), d['y']
    print(f'  shape={X.shape}  classi={list(np.unique(y))}')
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Separabilità per classe (boxplot + violinplot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_separability(X_sim, y_sim, X_real, y_real, out_dir: str):
    metrics_def = {
        'Combinata (amp_var + phase_diff_var)': motion_metric,
        'Ampiezza — var temporale':             amp_var_only,
        'Fase — var diff frame-to-frame':       phase_diff_var_only,
    }

    fig, axes = plt.subplots(len(metrics_def), 2,
                             figsize=(14, 4 * len(metrics_def)))
    fig.suptitle('Separabilità IDLE vs. attività — metriche CSI', fontsize=14)

    for row, (label, fn) in enumerate(metrics_def.items()):
        for col, (X, y, domain) in enumerate([
            (X_sim,  y_sim,  'Simulato'),
            (X_real, y_real, 'Reale'),
        ]):
            ax = axes[row, col]
            data   = [fn(X[y == cls]) for cls in CLASSES]
            colors = [CLASS_COLORS[cls] for cls in CLASSES]
            bp = ax.boxplot(data, patch_artist=True, labels=CLASSES,
                            medianprops=dict(color='black', linewidth=2))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_yscale('log')
            ax.set_ylabel(label if col == 0 else '')
            ax.set_title(f'{domain}' if row == 0 else '')
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'separability_boxplot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Salvato: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — ROC curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(X_sim, y_sim, X_real, y_real, out_dir: str):
    metrics_def = {
        'Combinata':  motion_metric,
        'Ampiezza':   amp_var_only,
        'Fase diff':  phase_diff_var_only,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ROC — rilevatore IDLE (IDLE=0, attività=1)', fontsize=13)

    for ax, (X, y, domain) in zip(axes, [
        (X_sim,  y_sim,  'Simulato'),
        (X_real, y_real, 'Reale'),
    ]):
        y_bin = (y != 'IDLE').astype(int)
        for label, fn in metrics_def.items():
            scores = fn(X)
            fpr, tpr, _ = roc_curve(y_bin, scores)
            auc = roc_auc_score(y_bin, scores)
            ax.plot(fpr, tpr, label=f'{label}  AUC={auc:.3f}', lw=2)

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('FPR (falsi allarmi)')
        ax.set_ylabel('TPR (detection rate)')
        ax.set_title(domain)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_idle_detector.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Salvato: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Simulazione pipeline real-time
# ══════════════════════════════════════════════════════════════════════════════

def simulate_pipeline(X_sim, y_sim, X_real, y_real, out_dir: str,
                      n_idle_warmup: int = 15, n_active: int = 25):
    """
    Costruisce una sequenza sintetica: N_IDLE campioni IDLE poi N_ACTIVE di attività.
    Esegue IdleDetector su ogni campione e plotta la metrica + decisione + soglia.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Simulazione pipeline real-time: IDLE → attività', fontsize=13)

    test_activities = ['WALK', 'FALL', 'RUN']

    for col, act in enumerate(test_activities):
        for row, (X, y, domain) in enumerate([
            (X_sim,  y_sim,  'Sim'),
            (X_real, y_real, 'Real'),
        ]):
            ax = axes[row, col]

            idle_pool = X[y == 'IDLE']
            act_pool  = X[y == act]
            if len(idle_pool) < n_idle_warmup or len(act_pool) < n_active:
                ax.set_title(f'{act} {domain} — dati insufficienti')
                continue

            sequence   = np.concatenate([idle_pool[:n_idle_warmup],
                                         act_pool[:n_active]], axis=0)
            true_labels = ['IDLE'] * n_idle_warmup + [act] * n_active

            detector   = IdleDetector(n_init=5, k_sigma=3.0)
            pred_idle  = []
            thresholds = []
            metric_vals = []

            for i, X_win in enumerate(sequence):
                m = detector._metric(X_win) if detector.threshold is not None else None
                is_idle, _ = detector.update(X_win)
                pred_idle.append(is_idle)
                thresholds.append(detector.threshold)
                mv = detector._metric(X_win)
                metric_vals.append(mv)

            t = np.arange(len(sequence))
            ax.semilogy(t, metric_vals, color='steelblue', lw=1.5, label='metrica')
            ax.semilogy(t, thresholds,  color='red',       lw=1.5,
                        linestyle='--', label='soglia adattiva')

            # Shading predizioni
            for i, (p, tl) in enumerate(zip(pred_idle, true_labels)):
                color = '#2ecc71' if p else '#e74c3c'
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=color)

            # Linea di transizione
            ax.axvline(n_idle_warmup - 0.5, color='black', lw=2, linestyle=':')
            ax.text(n_idle_warmup + 0.5, ax.get_ylim()[1] * 0.5,
                    f'← {act}', fontsize=9, color='black')

            n_correct = sum(
                (p and tl == 'IDLE') or (not p and tl != 'IDLE')
                for p, tl in zip(pred_idle, true_labels)
            )
            acc = n_correct / len(sequence) * 100

            ax.set_title(f'{act} — {domain}  (acc={acc:.0f}%)')
            ax.set_xlabel('Campione')
            ax.set_ylabel('Metrica (log)')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'pipeline_simulation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Salvato: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim',    default='dataset_v2',
                        help='Directory dataset simulato (default: dataset_v2)')
    parser.add_argument('--real',   default='dataset_real_v2',
                        help='Directory dataset reale (default: dataset_real_v2)')
    parser.add_argument('--no-roc', action='store_true',
                        help='Salta il plot ROC')
    parser.add_argument('--out',    default=OUT_DIR,
                        help=f'Directory output figure (default: {OUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X_sim,  y_sim  = load_dataset(os.path.join(args.sim,  'dataset_har_combined.npz'))
    X_real, y_real = load_dataset(os.path.join(args.real, 'dataset_har_combined.npz'))

    print('\n── Plot 1: separabilità per classe ──')
    plot_separability(X_sim, y_sim, X_real, y_real, args.out)

    if not args.no_roc:
        print('\n── Plot 2: ROC curve ──')
        plot_roc(X_sim, y_sim, X_real, y_real, args.out)

    print('\n── Plot 3: simulazione pipeline real-time ──')
    simulate_pipeline(X_sim, y_sim, X_real, y_real, args.out)

    print(f'\nTutte le figure salvate in {args.out}/')


if __name__ == '__main__':
    main()
