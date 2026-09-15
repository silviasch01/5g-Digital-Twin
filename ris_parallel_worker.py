import os
import numpy as np
from scipy.interpolate import interp1d

_MAT_COLORS = {
    'concrete': [.5, .5, .5], 'metal': [.75, .75, .75],
    'wood':     [.55, .27, .07], 'human': [.8, .6, .4],
}

def _get_material_props(freq_hz):
    f = freq_hz / 1e9
    return {
        'itu-concrete': {'eps': 1.0,  'sig': 0},
        'itu-brick':    {'eps': 3.75, 'sig': 0.038},
        'itu-glass':    {'eps': 6.27, 'sig': 0.0043 * f**1.1925},
        'itu-wood':     {'eps': 1.99, 'sig': 0.0047 * f**1.0718},
        'itu-metal':    {'eps': 1.0,  'sig': 1e7},
        'itu-human':    {'eps': 60.0, 'sig': 1.5},
        'metal':        {'eps': 1.0,  'sig': 1e7},
        'concrete':     {'eps': 5.31, 'sig': 0.0326 * f**0.8095},
        'wood':         {'eps': 1.99, 'sig': 0.0047 * f**1.0718},
        'marble':       {'eps': 7.0,  'sig': 0.02},
        'human':        {'eps': 60.0, 'sig': 1.5},
    }

def _apply_material_fix(scene, freq_hz):
    from sionna.rt import RadioMaterial
    fp  = _get_material_props(freq_hz)
    old = list(scene.radio_materials.keys())
    m2m = {}
    for name in old:
        obj  = scene.get(name)
        base = (name.replace('mat-itu_', '').replace('mat-itu-', '')
                    .replace('itu_', '').replace('itu-', '').replace('mat-', ''))
        lk   = name if name in fp else base
        if lk in fp:
            eps, sig = fp[lk]['eps'], fp[lk]['sig']
        else:
            try:
                eps = float(obj.relative_permittivity.numpy())
                sig = float(obj.conductivity.numpy())
            except Exception:
                eps, sig = 5.0, 0.01
        nm = RadioMaterial(f'{name}_fix', relative_permittivity=eps, conductivity=sig)
        if base in _MAT_COLORS:
            nm.color = _MAT_COLORS[base]
        scene.add(nm)
        m2m[name] = nm
    for o in scene.objects.values():
        if o.radio_material.name in m2m:
            o.radio_material = m2m[o.radio_material.name]
    for name in old:
        scene.remove(name)


def run_panels_on_gpu(gpu_id, panel_positions_chunk, n_ant,
                      person_ys, person_vels, n_steps,
                      person_x, person_z,
                      scene_path, freq, rx_pos,
                      csirs_mask_np, pilot_symbol_indices,
                      x_rg_np, fft_size, subcarrier_spacing, snr_db,
                      batch_size=32, max_depth=5, seed=41):
    """
    Calcola H_partial (n_steps, fft_size) per i panel assegnati su gpu_id.
    Chiamato in un subprocess spawn — importa tutto internamente.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                           PathSolver, subcarrier_frequencies)
    from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel

    device        = torch.device('cuda:0')
    sc_all        = np.arange(fft_size)
    no            = 1.0 / (10 ** (snr_db / 10))
    frequencies   = torch.tensor(
        subcarrier_frequencies(fft_size, subcarrier_spacing), dtype=torch.float32
    ).to(device)
    channel_apply = ApplyOFDMChannel(add_awgn=True)
    x_rg_t        = torch.tensor(x_rg_np, dtype=torch.complex64).to(device)

    scene = load_scene(scene_path, merge_shapes=False)
    _apply_material_fix(scene, freq)
    scene.frequency = freq
    scene.tx_array  = PlanarArray(num_rows=1, num_cols=1, pattern='dipole', polarization='V')
    scene.rx_array  = PlanarArray(num_rows=1, num_cols=1, pattern='dipole', polarization='V')
    scene.add(Receiver(name='RX', position=rx_pos))

    p_solver  = PathSolver()
    n_panels  = len(panel_positions_chunk)
    n_batches = (n_panels + batch_size - 1) // batch_size
    H_partial = np.zeros((n_steps, fft_size), dtype=np.complex64)

    for b_i, batch_start in enumerate(range(0, n_panels, batch_size)):
        batch_pos = panel_positions_chunk[batch_start:batch_start + batch_size]
        n_batch   = len(batch_pos)
        for idx, pos in enumerate(batch_pos):
            scene.add(Transmitter(name=f'TX_RIS_{idx:04d}', position=pos))

        for i in range(n_steps):
            py  = float(person_ys[i])
            vel = list(person_vels[i])
            scene.get('person').position = [person_x, py, person_z]
            scene.get('person').velocity = vel
            paths = p_solver(scene=scene, max_depth=max_depth, los=True,
                             specular_reflection=True, diffuse_reflection=True,
                             refraction=True, synthetic_array=True, seed=seed)
            a, tau = paths.cir(normalize_delays=False, out_type='numpy')
            for tx_i in range(n_batch):
                a_i   = a[:, :, tx_i:tx_i+1, :, :]
                if a_i.shape[3] > 1:
                    a_i = a_i.sum(axis=3, keepdims=True) / np.sqrt(n_ant)
                tau_i = (tau[:, tx_i:tx_i+1, :1, :] if tau.ndim == 4
                         else tau[:, tx_i:tx_i+1, :])
                a_t   = torch.tensor(a_i,   dtype=torch.complex64).unsqueeze(0).to(device)
                tau_t = torch.tensor(tau_i, dtype=torch.float32).unsqueeze(0).to(device)
                h_f   = cir_to_ofdm_channel(frequencies, a_t, tau_t, normalize=True)
                y_w   = channel_apply(x_rg_t, h_f, no)
                y_np  = y_w[0, 0, 0].cpu().numpy()
                tx_np = x_rg_t[0, 0, 0].cpu().numpy()
                H_parts = []
                for sym in pilot_symbol_indices:
                    m    = csirs_mask_np[sym]
                    H_ls = y_np[sym, m] / tx_np[sym, m]
                    sc_p = np.where(m)[0]
                    fa   = interp1d(sc_p, np.abs(H_ls), kind='linear', fill_value='extrapolate')
                    fp_  = interp1d(sc_p, np.unwrap(np.angle(H_ls)), kind='linear', fill_value='extrapolate')
                    H_parts.append(fa(sc_all) * np.exp(1j * fp_(sc_all)))
                H_partial[i] += np.mean(H_parts, axis=0).astype(np.complex64)

        for idx in range(n_batch):
            scene.remove(f'TX_RIS_{idx:04d}')
        torch.cuda.empty_cache()
        print(f'[GPU {gpu_id}] batch {b_i+1}/{n_batches} completato', flush=True)

    del scene, p_solver
    torch.cuda.empty_cache()
    print(f'[GPU {gpu_id}] DONE  H_partial.shape={H_partial.shape}', flush=True)
    return H_partial
