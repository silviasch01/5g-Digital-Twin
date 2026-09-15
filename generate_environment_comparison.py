"""
Generate the unknown_room floor plan and a same-scale comparison against
TTI-Lab (§4 System Model).

This is a companion script to generate_environment_figures.py (left
untouched) -- it does NOT regenerate figs/system_model/environment.{png,pdf},
it only adds:

  figs/system_model/environment_unknown_room.{png,pdf}  (new room, alone)
  figs/system_model/environment_comparison.{png,pdf}     (same-scale, side by side)

unknown_room is the "alien" room (15m x 3m x 4m) used to test generalization
to a genuinely unseen geometry -- see run_acquisition_parallel_v2.py
TX/RIS/RX_POS_UNKNOWN and indoor_envs/unknown_room_baseline/generate_furniture.py.

Each room is described by a physical, centre-origin coordinate config (see
TTILAB/UNKNOWN_ROOM below); draw_room() converts to a bottom-left-origin plot
coordinate system via (plot = physical + (room_w/2, room_h/2)) and renders
walls, partition, furniture, gNB/RIS/UE markers, and dimension annotations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGS_DIR = 'figs/system_model'
os.makedirs(FIGS_DIR, exist_ok=True)

matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})


# ═══════════════════════════════════════════════════════════════════════════
# Room configs (all coordinates are PHYSICAL, centre-origin, in metres)
# ═══════════════════════════════════════════════════════════════════════════

def _desk_chair(x0, y0, w, d, cw=0.45, cd=0.45, gap=0.10, facing=+1):
    """desk box (x0,y0,w,d) back-left corner + chair box in front, along x."""
    desk = (x0, x0 + w, y0, y0 + d)
    if facing > 0:
        cx0 = x0 + w + gap
    else:
        cx0 = x0 - gap - cw
    cy0 = y0 + (d - cd) / 2
    chair = (cx0, cx0 + cw, cy0, cy0 + cd)
    return desk, chair


# ── TTI-Lab (9.1m x 4.0m) ─────────────────────────────────────────────────────
_ttilab_desks_raw = [
    (-4.00, -1.70), (-4.00, -0.90), (-4.00, -0.10), (-4.00, 0.70),
    (-1.50, -1.70), (-1.50, -0.90), (-1.50, -0.10), (-1.50, 0.70),
    (2.50, -1.50), (2.50, -0.50), (2.50, 0.50),
]
_ttilab_desks, _ttilab_chairs = [], []
for dx, dy in _ttilab_desks_raw:
    d, c = _desk_chair(dx, dy, w=0.60, d=1.00, facing=+1)
    _ttilab_desks.append(d)
    _ttilab_chairs.append(c)

TTILAB = dict(
    label='Target Environment',
    room_w=9.10, room_h=4.00,
    wall_phys=(1.45, 1.65, -2.00, 1.00),
    tx_phys=(-1.00, 1.89),
    ris_phys=(4.40, 1.00),
    rx_los_phys=(0.50, 0.00),
    rx_nlos_phys=(4.00, -1.00),
    rx_grid_phys=[(px, py) for px in np.arange(-4, 5, 1)
                  for py in (-1.5, -0.5, 0.5, 1.5)],
    desks_phys=_ttilab_desks,
    chairs_phys=_ttilab_chairs,
    gnb_label_below=False,   # gNB label above the marker
    ue_label_lift=0.06,      # nudge UE (LOS)/(NLOS) labels up, closer to their marker
)

# ── unknown_room (15m x 3m x 4m, "alien" room) ────────────────────────────────
_unk_desk_specs = [
    (-7.3, -1.3, +1), (-7.3, 0.3, +1), (-6.4, -1.3, +1), (-6.4, 0.3, +1),
    (6.7, -1.3, -1), (6.7, 0.3, -1),
]
_unk_desks, _unk_chairs = [], []
for dx, dy, facing in _unk_desk_specs:
    d, c = _desk_chair(dx, dy, w=0.60, d=1.00, facing=facing)
    _unk_desks.append(d)
    _unk_chairs.append(c)

UNKNOWN_ROOM = dict(
    label='Cross-Validation Environment',
    room_w=15.0, room_h=3.0,
    wall_phys=(2.5, 2.7, -1.5, 0.5),
    tx_phys=(-1.5, 1.2),
    ris_phys=(7.3, 0.8),
    rx_los_phys=(1.0, -1.0),
    rx_nlos_phys=(6.0, -1.0),
    rx_grid_phys=None,   # fixed-point baseline/target dataset, not a training grid
    desks_phys=_unk_desks,
    chairs_phys=_unk_chairs,
    gnb_label_below=True,    # gNB label below the marker
    ue_label_lift=0.0,       # keep UE (LOS)/(NLOS) labels as-is
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared drawing routine
# ═══════════════════════════════════════════════════════════════════════════

def draw_room(ax, cfg, show_dims=True, show_legend=True, inch_per_m=1.2):
    room_w, room_h = cfg['room_w'], cfg['room_h']
    ox, oy = room_w / 2, room_h / 2

    # Margins/annotation offsets below are fixed in *inches*, then converted
    # to data-metres via inch_per_m, so they render at a constant absolute
    # size regardless of the panel's own scale. draw_room() is reused at very
    # different inch_per_m values (~1.2 for the standalone unknown_room
    # figure vs. a smaller shared scale for the same-scale comparison
    # figure); a fixed metre-offset would shrink to nothing at the smaller
    # scale, causing the dimension labels to visually collide with axis
    # ticks/labels and spines.
    left_pad    = 0.55 / inch_per_m
    right_pad   = 1.35 / inch_per_m
    bottom_pad  = 0.55 / inch_per_m
    top_pad     = 0.70 / inch_per_m
    dim_off     = 0.24 / inch_per_m   # dimension arrow distance from room edge
    dim_txt_gap = 0.11 / inch_per_m   # extra gap from arrow to its text

    # gNB/UE markers and labels are sized in points/pt-fontsize, which do NOT
    # auto-scale with inch_per_m the way patches drawn in data (metre) units
    # do. At the comparison figure's smaller shared scale (INCH_PER_M=0.6 vs.
    # the ~1.2 used for the standalone figure) the same fixed sizes used to
    # look oversized and collide with furniture/RIS/each other. mk_scale
    # shrinks them down at smaller scales (never grows them past the
    # standalone look), with a floor so labels stay legible.
    mk_scale    = min(1.0, inch_per_m / 1.2)
    gnb_ms      = max(14, 26 * mk_scale)
    ue_ms       = max(10, 18 * mk_scale)
    lbl_fs      = max(9, 12 * mk_scale)
    # Text-to-marker gap, fixed in inches (same trick as dim_off above) so it
    # doesn't shrink to nothing -- and the label collide with the marker --
    # at smaller scales.
    gnb_txt_off = 0.14 / inch_per_m
    ue_txt_off  = 0.16 / inch_per_m

    def p(phys_x, phys_y):
        return phys_x + ox, phys_y + oy

    def furn(x0, x1, y0, y1):
        return (x0 + ox, x1 + ox, y0 + oy, y1 + oy)

    tx_xy = p(*cfg['tx_phys'])
    ris_xy = p(*cfg['ris_phys'])
    rx_los_xy = p(*cfg['rx_los_phys'])
    rx_nlos_xy = p(*cfg['rx_nlos_phys'])
    wx0, wx1, wy0, wy1 = cfg['wall_phys']
    wall = (wx0 + ox, wx1 + ox, wy0 + oy, wy1 + oy)

    ax.set_aspect('equal')
    ax.set_xlim(-left_pad, room_w + right_pad)
    ax.set_ylim(-bottom_pad, room_h + top_pad)

    # Room outline
    ax.add_patch(mpatches.Rectangle(
        (0, 0), room_w, room_h,
        linewidth=2.5, edgecolor='#111111', facecolor='#f5f5f0', zorder=0))

    # Dividing wall
    ax.add_patch(mpatches.Rectangle(
        (wall[0], wall[2]), wall[1] - wall[0], wall[3] - wall[2],
        linewidth=1.0, edgecolor='#111111', facecolor='#333333', zorder=2))

    # Furniture
    for (x0, x1, y0, y1) in [furn(*d) for d in cfg['desks_phys']]:
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0.8, edgecolor='#6b3a1f', facecolor='#c8894a', zorder=3))
    for (x0, x1, y0, y1) in [furn(*c) for c in cfg['chairs_phys']]:
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0.6, edgecolor='#4a2a0f', facecolor='#7a4a1e', zorder=3))

    # RX simulation grid (only where one exists)
    if cfg['rx_grid_phys']:
        for (px, py) in cfg['rx_grid_phys']:
            gx, gy = p(px, py)
            ax.plot(gx, gy, marker='s', markersize=9, color='#4a90d9',
                    markeredgecolor='#2060a0', markeredgewidth=0.7, zorder=5)

    # # RIS bar on east wall
    # ax.add_patch(mpatches.FancyBboxPatch(
    #     (room_w - 0.10, ris_xy[1] - 0.38), 0.10, 0.76,
    #     boxstyle='square,pad=0.0',
    #     linewidth=2, edgecolor='#922b21', facecolor='#e74c3c', zorder=6))
    # ax.text(room_w + 0.15, ris_xy[1], 'RIS',
    #         ha='left', va='center', fontsize=lbl_fs, fontweight='bold',
    #         color='#922b21', zorder=7)

    # gNB (TX) -- label goes above the marker by default, below it for rooms
    # with gnb_label_below=True (currently just the cross-validation room, to
    # avoid the label running into the top wall/margin).
    gnb_below = cfg.get('gnb_label_below', False)
    gnb_sign  = -1 if gnb_below else 1
    ax.plot(tx_xy[0], tx_xy[1], marker='^', markersize=gnb_ms,
            color='#1a6fa8', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
    ax.text(tx_xy[0], tx_xy[1] + gnb_sign * gnb_txt_off, 'gNB',
            ha='center', va=('top' if gnb_below else 'bottom'), fontsize=lbl_fs,
            fontweight='bold', color='#1a6fa8', zorder=9)

    # UE labels sit below their marker; ue_label_lift nudges them closer to it.
    ue_txt_off_local = ue_txt_off - cfg.get('ue_label_lift', 0.0)

    # UE LOS
    ax.plot(rx_los_xy[0], rx_los_xy[1], marker='D', markersize=ue_ms,
            color='#1e8449', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
    ax.text(rx_los_xy[0], rx_los_xy[1] - ue_txt_off_local, 'UE (LOS)',
            ha='center', va='top', fontsize=lbl_fs, fontweight='bold',
            color='#1e8449', zorder=9)

    # UE NLOS
    ax.plot(rx_nlos_xy[0], rx_nlos_xy[1], marker='D', markersize=ue_ms,
            color='#e67e22', zorder=8, markeredgecolor='white', markeredgewidth=0.8)
    ax.text(rx_nlos_xy[0], rx_nlos_xy[1] - ue_txt_off_local, 'UE (NLOS)',
            ha='center', va='top', fontsize=lbl_fs, fontweight='bold',
            color='#e67e22', zorder=9)

    if show_dims:
        akw = dict(arrowstyle='<->', color='#333333', lw=1.2,
                   mutation_scale=11, shrinkA=0, shrinkB=0)
        yd = -dim_off
        ax.annotate('', xy=(room_w, yd), xytext=(0, yd), arrowprops=akw)
        ax.text(room_w / 2, yd - dim_txt_gap, f'$\\approx {room_w:.1f}\\,\\mathrm{{m}}$',
                ha='center', va='top', fontsize=12, color='#333333')
        xd = -dim_off
        ax.annotate('', xy=(xd, room_h), xytext=(xd, 0), arrowprops=akw)
        ax.text(xd - dim_txt_gap, room_h / 2, f'$\\approx {room_h:.1f}\\,\\mathrm{{m}}$',
                ha='right', va='center', fontsize=12, color='#333333', rotation=90)

    ax.set_xlabel('$x$ [m]', fontsize=13, labelpad=6)
    ax.set_ylabel('$y$ [m]', fontsize=13, labelpad=6)
    ax.tick_params(direction='out', pad=4)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.set_title(cfg['label'], fontsize=14, fontweight='bold')

    if show_legend:
        legend_handles = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#1a6fa8',
                       markeredgecolor='white', markersize=12, label='gNB (TX)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#1e8449',
                       markeredgecolor='white', markersize=9, label='UE (LOS)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e67e22',
                       markeredgecolor='white', markersize=9, label='UE (NLOS)'),
            mpatches.Patch(facecolor='#e74c3c', edgecolor='#922b21', label='RIS 4×4'),
        ]
        if cfg['rx_grid_phys']:
            legend_handles.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4a90d9',
                           markeredgecolor='#2060a0', markersize=8, label='Sim. RX grid'))
        legend_handles += [
            mpatches.Patch(facecolor='#c8894a', edgecolor='#6b3a1f', label='Desks'),
            mpatches.Patch(facecolor='#7a4a1e', edgecolor='#4a2a0f', label='Chairs'),
            mpatches.Patch(facecolor='#333333', edgecolor='#111111', label='Dividing wall'),
        ]
        ax.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.22), ncol=4,
                  fontsize=11, framealpha=0.95, edgecolor='#ccc',
                  handlelength=1.4, columnspacing=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. unknown_room figure (standalone)
# ═══════════════════════════════════════════════════════════════════════════

FIG_W2 = 16.0
FIG_H2 = FIG_W2 * (UNKNOWN_ROOM['room_h'] / UNKNOWN_ROOM['room_w']) + 1.4
fig, ax = plt.subplots(figsize=(FIG_W2, FIG_H2))
draw_room(ax, UNKNOWN_ROOM)
plt.tight_layout()
for ext in ('png', 'pdf'):
    out = f'{FIGS_DIR}/environment_unknown_room.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Side-by-side comparison, SAME metres-per-inch scale on both panels
#    (so the shape difference -- short+deep vs. long+narrow -- is real,
#    not an artefact of each panel being auto-scaled to fill its own axes)
#
#    NOTE: plt.subplots(2,1) + set_aspect('equal') is NOT enough to guarantee
#    a shared scale: matplotlib fits each axes' data aspect ratio into an
#    *identically shaped* subplot cell, so the two rooms end up rendered at
#    different inches-per-metre even though a single INCH_PER_M is "shared"
#    in name. Fix: size each axes box in absolute inches from its own room's
#    data span, via fig.add_axes() with an explicit rect, so 1m maps to the
#    same number of inches in both panels regardless of how matplotlib would
#    otherwise auto-fit them.
# ═══════════════════════════════════════════════════════════════════════════

INCH_PER_M = 0.6   # shared scale for both panels
# draw_room()'s xlim/ylim padding (left_pad+right_pad, bottom_pad+top_pad) is
# fixed in *inches* (see draw_room), so it contributes a constant number of
# inches to the axes box regardless of INCH_PER_M -- not a metre padding.
LR_PAD_IN, TB_PAD_IN = 0.55 + 1.35, 0.55 + 0.70

def _axes_size_in(cfg):
    return cfg['room_w'] * INCH_PER_M + LR_PAD_IN, cfg['room_h'] * INCH_PER_M + TB_PAD_IN

axw1, axh1 = _axes_size_in(TTILAB)
axw2, axh2 = _axes_size_in(UNKNOWN_ROOM)

left_in     = 0.70   # y-axis label/ticks
right_in    = 0.45   # safety margin
title_in    = 0.45   # per-panel title
xlabel_in   = 0.55   # per-panel x-label/ticks
gap_in      = 0.30   # breathing room between the two stacked panels
legend_in   = 1.00   # shared legend at the very bottom

fig_w = left_in + max(axw1, axw2) + right_in
fig_h = (title_in + axh1 + xlabel_in + gap_in
         + title_in + axh2 + xlabel_in + legend_in)

fig = plt.figure(figsize=(fig_w, fig_h))

# Bottom panel: unknown_room (leave room below its box for its own x-label,
# above the shared legend)
bottom2 = legend_in + xlabel_in
ax2 = fig.add_axes([left_in / fig_w, bottom2 / fig_h, axw2 / fig_w, axh2 / fig_h])

# Top panel: TTI-Lab, stacked above with a gap (and its own x-label reserved
# just below its box, same as ax2)
bottom1 = bottom2 + axh2 + title_in + gap_in + xlabel_in
ax1 = fig.add_axes([left_in / fig_w, bottom1 / fig_h, axw1 / fig_w, axh1 / fig_h])

for ax, cfg in zip((ax1, ax2), (TTILAB, UNKNOWN_ROOM)):
    draw_room(ax, cfg, show_legend=False, inch_per_m=INCH_PER_M)

handles = [
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#1a6fa8',
               markeredgecolor='white', markersize=12, label='gNB (TX)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#1e8449',
               markeredgecolor='white', markersize=9, label='UE (LOS)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e67e22',
               markeredgecolor='white', markersize=9, label='UE (NLOS)'),
    #mpatches.Patch(facecolor='#e74c3c', edgecolor='#922b21', label='RIS 4×4'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4a90d9',
               markeredgecolor='#2060a0', markersize=8, label='Sim. RX grid'),
    mpatches.Patch(facecolor='#c8894a', edgecolor='#6b3a1f', label='Desks'),
    mpatches.Patch(facecolor='#7a4a1e', edgecolor='#4a2a0f', label='Chairs'),
    mpatches.Patch(facecolor='#333333', edgecolor='#111111', label='Dividing wall'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11,
           framealpha=0.95, edgecolor='#ccc', handlelength=1.4,
           columnspacing=1.0, bbox_to_anchor=(0.5, 0.30 * legend_in / fig_h))
for ext in ('png', 'pdf'):
    out = f'{FIGS_DIR}/environment_comparison.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close(fig)

print('Done.')
