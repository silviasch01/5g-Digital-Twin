"""
Generate annotated floor plan for the paper (§4 System Model).

Origin at bottom-left corner of the room:
  x ∈ [0, 9.1] m   (east)
  y ∈ [0, 4.0] m   (north)

Physical → plot coordinate transform: plot = physical + (4.55, 2.00)

Key positions (plot coords):
  gNB  : (3.55, 3.89)
  RIS  : (8.95, 3.00)  east wall
  RX0  : (8.55, 1.00)  real UE
  RX grid: x ∈ {0.55,1.55,...,8.55} × y ∈ {0.5,1.5,2.5,3.5}  (9×4=36)

Outputs:
  figs/system_model/environment.{png,pdf}
  paper-src/.../figures/system_model/environment.{png,pdf}
"""

import os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGS_DIR  = 'figs/system_model'
PAPER_DIR = 'paper-src/CNIT-2026-technical-report-chapter/figures/system_model'
os.makedirs(FIGS_DIR,  exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

# ── Coordinate transform: physical (centre-origin) → plot (BL-origin) ────────
OX, OY = 4.55, 2.00   # offset

def p(phys_x, phys_y):
    return phys_x + OX, phys_y + OY

# ── Room in plot coords ───────────────────────────────────────────────────────
ROOM_W, ROOM_H = 9.10, 4.00          # metres
ROOM_X = (0.0, ROOM_W)
ROOM_Y = (0.0, ROOM_H)

# ── Key nodes (plot coords) ───────────────────────────────────────────────────
TX_XY  = p(-1.00,  1.89)    # gNB  → (3.55, 3.89)
RIS_XY = p( 4.40,  1.00)    # RIS  → (8.95, 3.00)
RX_LOS_XY  = p( 0.50,  0.00)   # UE LOS  (dataset_real_v4) → (5.05, 2.00)
RX_NLOS_XY = p( 4.00, -1.00)   # UE NLOS (dataset_real_v2) → (8.55, 1.00) — behind dividing wall

# ── Simulation RX grid (plot coords) ─────────────────────────────────────────
RX_X_phys = np.arange(-4, 5, 1)
RX_Y_phys = np.array([-1.5, -0.5, 0.5, 1.5])
RX_PTS = [(px + OX, py + OY) for px in RX_X_phys for py in RX_Y_phys]

# ── Dividing wall (plot coords): physical x=[1.45,1.65], y=[-2,1] ────────────
WALL_X0, WALL_X1 = 1.45 + OX, 1.65 + OX   # 6.00, 6.20
WALL_Y0, WALL_Y1 = -2.00 + OY, 1.00 + OY  # 0.00, 3.00

# ── Furniture bounding boxes (plot coords) ────────────────────────────────────
def furn(x0, x1, y0, y1):
    return (x0 + OX, x1 + OX, y0 + OY, y1 + OY)

DESKS = [
    furn(-4.00, -3.40, -1.70, -0.70), furn(-4.00, -3.40, -0.90,  0.10),
    furn(-4.00, -3.40, -0.10,  0.90), furn(-4.00, -3.40,  0.70,  1.70),
    furn(-1.50, -0.90, -1.70, -0.70), furn(-1.50, -0.90, -0.90,  0.10),
    furn(-1.50, -0.90, -0.10,  0.90), furn(-1.50, -0.90,  0.70,  1.70),
    furn( 2.50,  3.10, -1.50, -0.50), furn( 2.50,  3.10, -0.50,  0.50),
    furn( 2.50,  3.10,  0.50,  1.50),
]
CHAIRS = [
    furn(-3.30, -2.85, -1.42, -0.98), furn(-3.30, -2.85, -0.62, -0.17),
    furn(-3.30, -2.85,  0.17,  0.62), furn(-3.30, -2.85,  0.98,  1.42),
    furn(-0.80, -0.35, -1.42, -0.98), furn(-0.80, -0.35, -0.62, -0.17),
    furn(-0.80, -0.35,  0.17,  0.62), furn(-0.80, -0.35,  0.98,  1.42),
    furn( 3.20,  3.65, -1.23, -0.77), furn( 3.20,  3.65, -0.22,  0.22),
    furn( 3.20,  3.65,  0.77,  1.23),
]

# ── Figure ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})

FIG_W = 14.0
FIG_H = FIG_W * (ROOM_H / ROOM_W) + 1.0   # extra for legend

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_aspect('equal')
ax.set_xlim(-0.45, ROOM_X[1] + 1.15)  # extra right for RIS label
ax.set_ylim(-0.45, ROOM_Y[1] + 0.60)  # extra top for gNB label

# ── Room outline ──────────────────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle(
    (ROOM_X[0], ROOM_Y[0]), ROOM_W, ROOM_H,
    linewidth=2.5, edgecolor='#111111', facecolor='#f5f5f0', zorder=0))

# ── Dividing wall ─────────────────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle(
    (WALL_X0, WALL_Y0), WALL_X1 - WALL_X0, WALL_Y1 - WALL_Y0,
    linewidth=1.0, edgecolor='#111111', facecolor='#333333', zorder=2))

# ── Furniture ─────────────────────────────────────────────────────────────────
for (x0, x1, y0, y1) in DESKS:
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        linewidth=0.8, edgecolor='#6b3a1f', facecolor='#c8894a', zorder=3))
for (x0, x1, y0, y1) in CHAIRS:
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        linewidth=0.6, edgecolor='#4a2a0f', facecolor='#7a4a1e', zorder=3))

# ── RX simulation grid ────────────────────────────────────────────────────────
for (gx, gy) in RX_PTS:
    ax.plot(gx, gy, marker='s', markersize=9, color='#4a90d9',
            markeredgecolor='#2060a0', markeredgewidth=0.7, zorder=5)

# ── RIS bar on east wall ──────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (ROOM_X[1]-0.10, RIS_XY[1]-0.38), 0.10, 0.76,
    boxstyle='square,pad=0.0',
    linewidth=2, edgecolor='#922b21', facecolor='#e74c3c', zorder=6))
# Label to the RIGHT of the east wall, outside the room
ax.text(ROOM_X[1] + 0.15, RIS_XY[1],
        f'RIS\n({RIS_XY[0]:.2f}, {RIS_XY[1]:.2f})',
        ha='left', va='center', fontsize=12, fontweight='bold',
        color='#922b21', zorder=7)

# ── gNB (TX) — triangle marker ────────────────────────────────────────────────
ax.plot(TX_XY[0], TX_XY[1], marker='^', markersize=26,
        color='#1a6fa8', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
ax.text(TX_XY[0], TX_XY[1] + 0.25,
        f'gNB\n({TX_XY[0]:.2f}, {TX_XY[1]:.2f})',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#1a6fa8', zorder=9)

# ── UE LOS ───────────────────────────────────────────────────────────────────
ax.plot(RX_LOS_XY[0], RX_LOS_XY[1], marker='D', markersize=18,
        color='#1e8449', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
ax.text(RX_LOS_XY[0], RX_LOS_XY[1] - 0.28, 'UE (LOS)',
        ha='center', va='top', fontsize=12, fontweight='bold',
        color='#1e8449', zorder=9)

# ── UE NLOS ──────────────────────────────────────────────────────────────────
ax.plot(RX_NLOS_XY[0], RX_NLOS_XY[1], marker='D', markersize=18,
        color='#e67e22', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
ax.text(RX_NLOS_XY[0], RX_NLOS_XY[1] - 0.28, 'UE (NLOS)',
        ha='center', va='top', fontsize=12, fontweight='bold',
        color='#e67e22', zorder=9)

# ── Dimension annotations ─────────────────────────────────────────────────────
akw = dict(arrowstyle='<->', color='#333333', lw=1.2,
           mutation_scale=11, shrinkA=0, shrinkB=0)
yd = ROOM_Y[0] - 0.20
ax.annotate('', xy=(ROOM_X[1], yd), xytext=(ROOM_X[0], yd), arrowprops=akw)
ax.text(ROOM_W/2, yd - 0.07, f'$\\approx {ROOM_W:.1f}\\,\\mathrm{{m}}$',
        ha='center', va='top', fontsize=12, color='#333333')
xd = ROOM_X[0] - 0.20
ax.annotate('', xy=(xd, ROOM_Y[1]), xytext=(xd, ROOM_Y[0]), arrowprops=akw)
ax.text(xd - 0.07, ROOM_H/2, f'$\\approx {ROOM_H:.1f}\\,\\mathrm{{m}}$',
        ha='right', va='center', fontsize=12, color='#333333', rotation=90)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('$x$ [m]', fontsize=13, labelpad=6)
ax.set_ylabel('$y$ [m]', fontsize=13, labelpad=6)
ax.set_xticks(range(0, 10))
ax.set_xticklabels([str(v) for v in range(0, 10)], fontsize=12)
ax.set_yticks(range(0, 5))
ax.set_yticklabels([str(v) for v in range(0, 5)], fontsize=12)
ax.tick_params(direction='out', pad=4)
ax.grid(True, alpha=0.18, linewidth=0.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#1a6fa8',
               markeredgecolor='white', markersize=12, label='gNB (TX)'),
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#1e8449',
               markeredgecolor='white', markersize=9,  label='UE (LOS)'),
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#e67e22',
               markeredgecolor='white', markersize=9,  label='UE (NLOS)'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='#922b21', label='RIS 4×4'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#4a90d9',
               markeredgecolor='#2060a0', markersize=8, label='Sim. RX grid (36 pts)'),
    mpatches.Patch(facecolor='#c8894a', edgecolor='#6b3a1f', label='Desks'),
    mpatches.Patch(facecolor='#7a4a1e', edgecolor='#4a2a0f', label='Chairs'),
    mpatches.Patch(facecolor='#333333', edgecolor='#111111', label='Dividing wall'),
]
ax.legend(handles=legend_handles, loc='lower center',
          bbox_to_anchor=(0.5, -0.18), ncol=8,
          fontsize=12, framealpha=0.95, edgecolor='#ccc',
          handlelength=1.4, columnspacing=1.0)

plt.tight_layout()

for ext in ('png', 'pdf'):
    out = f'{FIGS_DIR}/environment.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    shutil.copy(out, f'{PAPER_DIR}/environment.{ext}')
    print(f'Saved: {out}  →  {PAPER_DIR}/')
plt.close()
print('Done.')
