"""
Latinus et al. 2015 ERP replication — non-social task
Figures 3, 4, 5: Grand average ERPs (away vs toward) at LH and RH N170 clusters

Cluster definition (paper quote):
  "N170 latencies and amplitudes were measured from the ERPs averaged over a
   nine-electrode cluster (Figure 1C), centred on the electrode where the grand
   average (collapsed for conditions) was maximal between 142 and 272 ms
   post-stimulus."  (Latinus et al. 2015, Methods)
Cluster defined separately for LH and RH; same cluster applied to all subjects.

Bootstrap 95% CI (paper quote):
  "Shaded areas represent the 95% confidence interval built using bootstrap
   (n=1000) with replacement of the data under H1."  (Latinus et al. 2015,
   Figure captions 3-5)
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = DATA_DIR
N_BOOTSTRAP = 1000
SEED        = 42
rng         = np.random.default_rng(SEED)

# EGI system delay — paper quote:
# "An advisory notice from the EGI EEG system manufacturer has informed us about
#  an 18 ms delay between real-time acquisition (to which events are synchronized)
#  and the EEG signal. Consequently, a post hoc latency factor of 18 ms was
#  applied to all ERP latencies."  (Latinus et al. 2015, footnote 1)
#
# t=0 in our epochs = trigger time; stimulus appeared 18ms later on screen.
# The N170 dip therefore appears ~18ms earlier in our plots than the paper's
# reported latencies (paper reports 213-222ms -> dip in data at ~195-204ms).
# This does not affect the away-vs-toward amplitude difference we are measuring.
EGI_DELAY_S = 0.018  # for reference only — data is not shifted

# Cluster search window — paper quote:
# "N170 latencies and amplitudes were measured from the ERPs averaged over a
#  nine-electrode cluster (Figure 1C), centred on the electrode where the grand
#  average (collapsed for conditions) was maximal between 142 and 272 ms
#  post-stimulus."  (Latinus et al. 2015, Methods)
# Window is in trigger-locked data time, consistent with the paper's definition.
CLUSTER_TMIN = 0.142
CLUSTER_TMAX = 0.272
N_CLUSTER    = 9

# Fixed N170 clusters from make_ica_report.py (GSN-HydroCel-257 occipito-temporal)
FIXED_RH = ['E158', 'E150', 'E159', 'E140', 'E151', 'E160', 'E168', 'E152', 'E169']
FIXED_LH = ['E123', 'E116', 'E115', 'E109', 'E108', 'E107', 'E114', 'E98',  'E106']

# Condition groups — event IDs in the Brainlife epoch files (non-social task only)
# Away conditions: gaze moves away from observer
# Toward conditions: gaze moves toward observer
COND_GROUPS = {
    'fig3': {
        'title':        'Full transitions',
        'away':         [13, 14],   # D_REA, D_LEA  — Dir→Ext
        'toward':       [15, 16],   # REA_D, LEA_D  — Ext→Dir
        'away_label':   'Dir-Ext',
        'toward_label': 'Ext-Dir',
    },
    'fig4': {
        'title':        'Intermediate-to-endpoint',
        'away':         [17, 18],   # RIA_REA, LIA_LEA  — Int→Ext
        'toward':       [19, 20],   # RIA_D, LIA_D      — Int→Dir
        'away_label':   'Int-Ext',
        'toward_label': 'Int-Dir',
    },
    'fig5': {
        'title':        'Endpoint-to-intermediate',
        'away':         [21, 22],   # D_RIA, D_LIA      — Dir→Int
        'toward':       [23, 24],   # REA_RIA, LEA_LIA  — Ext→Int
        'away_label':   'Dir-Int',
        'toward_label': 'Ext-Int',
    },
}
ALL_CIDS = [c for g in COND_GROUPS.values() for c in g['away'] + g['toward']]

# ── LOAD EPOCHS ───────────────────────────────────────────────────────────────
fif_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'meg-epo.fif')))
print(f'Found {len(fif_files)} epoch files')

subjects   = []
all_evoked = {}   # sub → {cid: mne.Evoked}

for fif in fif_files:
    sub = os.path.basename(os.path.dirname(fif))
    try:
        epochs = mne.read_epochs(fif, verbose=False, preload=True)
    except Exception as e:
        print(f'  {sub}: SKIP — {e}')
        continue

    available = set(epochs.event_id.values())
    needed    = set(ALL_CIDS)
    if not needed.issubset(available):
        print(f'  {sub}: SKIP — missing events {needed - available}')
        continue

    if epochs.info['bads']:
        print(f'  {sub}: interpolating {len(epochs.info["bads"])} bad channels: {epochs.info["bads"]}')
        epochs.interpolate_bads(reset_bads=True)

    # Average epochs per condition
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    sub_evoked = {}
    for cid in ALL_CIDS:
        sub_evoked[cid] = epochs[id_to_name[cid]].average()

    all_evoked[sub] = sub_evoked
    subjects.append(sub)
    print(f'  {sub}: {len(epochs)} epochs, {len(epochs.ch_names)} channels')

print(f'\nIncluded subjects: {len(subjects)}: {subjects}')
if not subjects:
    raise RuntimeError('No valid subjects found — check DATA_DIR and downloads')

# ── GRAND AVERAGE ACROSS ALL SUBJECTS AND CONDITIONS ─────────────────────────
# Used only for cluster definition
all_evo_list = [all_evoked[sub][cid]
                for sub in subjects
                for cid in ALL_CIDS]
grand_avg = mne.grand_average(all_evo_list, drop_bads=False)

# ── DEFINE N170 CLUSTERS ──────────────────────────────────────────────────────
picks_eeg = mne.pick_types(grand_avg.info, eeg=True, exclude='bads')
ch_names  = [grand_avg.ch_names[i] for i in picks_eeg]
pos3d     = np.array([grand_avg.info['chs'][i]['loc'][:3] for i in picks_eeg])

# Check we have valid positions
if np.all(pos3d == 0):
    raise RuntimeError('Electrode positions are all zero — montage not set in the epoch files')

# Split hemispheres by x-coordinate (MNE head frame: +x = right, −x = left)
# Restrict to posterior electrodes (y < -0.01) so the search stays in the
# occipito-temporal region where N170 lives — prevents spurious frontal peaks.
posterior_mask = pos3d[:, 1] < -0.01
lh_mask = (pos3d[:, 0] < -0.005) & posterior_mask
rh_mask = (pos3d[:, 0] >  0.005) & posterior_mask

times  = grand_avg.times
tmask  = (times >= CLUSTER_TMIN) & (times <= CLUSTER_TMAX)
data   = grand_avg.data[picks_eeg, :]   # (n_eeg_ch, n_times)

def find_cluster(data, tmask, hemi_mask, pos3d, n=N_CLUSTER):
    hemi_idx   = np.where(hemi_mask)[0]
    mean_amp   = data[hemi_idx][:, tmask].mean(axis=1)
    peak_local = np.argmin(mean_amp)          # most negative = N170
    peak_idx   = hemi_idx[peak_local]
    dists      = np.linalg.norm(pos3d - pos3d[peak_idx], axis=1)
    cluster    = np.argsort(dists)[:n]
    return cluster, peak_idx

lh_cluster, lh_peak = find_cluster(data, tmask, lh_mask, pos3d)
rh_cluster, rh_peak = find_cluster(data, tmask, rh_mask, pos3d)

# Store clusters as channel names — safe to use across different evokeds
# (grand_avg may have fewer channels than individual evokeds after common-channel
# detection, so numeric indices from grand_avg cannot be reused on subject evokeds)
lh_cluster_names = [ch_names[i] for i in lh_cluster]
rh_cluster_names = [ch_names[i] for i in rh_cluster]

print(f'\nLH cluster centre: {ch_names[lh_peak]}')
print(f'  electrodes: {lh_cluster_names}')
print(f'RH cluster centre: {ch_names[rh_peak]}')
print(f'  electrodes: {rh_cluster_names}')

times_ms = times * 1000  # used throughout all plots below

# ── TOPOMAP: grand average at N170 peak + cluster locations ──────────────────
# Three panels: topomap at peak time, LH cluster highlighted, RH cluster highlighted
# Use RH cluster mean (not all-channel average) to find N170 peak time for topomap
rh_cluster_signal = grand_avg.data[picks_eeg[rh_cluster], :][:, tmask].mean(axis=0)
peak_time = times[tmask][np.argmin(rh_cluster_signal)]
print(f'\nN170 peak time at RH cluster: {peak_time*1000:.0f} ms')

fig_topo, axes_t = plt.subplots(1, 3, figsize=(13, 4))

# Panel 1: full scalp topomap at N170 peak, no highlights
grand_avg.plot_topomap(times=peak_time, axes=axes_t[0], show=False,
                       colorbar=False, time_unit='ms')
axes_t[0].set_title(f'Grand avg topomap\nat {peak_time*1000:.0f} ms (N170 peak)', fontsize=9)

# Panel 2: LH cluster highlighted
lh_ch_names = [ch_names[i] for i in lh_cluster]
grand_avg.plot_topomap(times=peak_time, axes=axes_t[1], show=False,
                       colorbar=False, time_unit='ms')
# Overlay cluster electrode markers
for ch in lh_ch_names:
    idx = grand_avg.ch_names.index(ch)
    xy = grand_avg.info['chs'][idx]['loc'][:2]
axes_t[1].set_title(f'LH cluster  ({ch_names[lh_peak]})\n{lh_ch_names}', fontsize=7)

# Panel 3: RH cluster highlighted
rh_ch_names = [ch_names[i] for i in rh_cluster]
grand_avg.plot_topomap(times=peak_time, axes=axes_t[2], show=False,
                       colorbar=False, time_unit='ms')
axes_t[2].set_title(f'RH cluster  ({ch_names[rh_peak]})\n{rh_ch_names}', fontsize=7)

# Highlight cluster electrodes using MNE's mask parameter
# mask must be 2D: (n_channels, n_times) — one time point here so shape (n_ch, 1)
for ax, cluster_idx in [(axes_t[1], lh_cluster), (axes_t[2], rh_cluster)]:
    mask = np.zeros((len(grand_avg.ch_names), len(grand_avg.times)), dtype=bool)
    for i in cluster_idx:
        mask[picks_eeg[i], :] = True
    ax.clear()
    grand_avg.plot_topomap(times=peak_time, axes=ax, show=False, colorbar=False,
                           time_unit='ms', mask=mask,
                           mask_params=dict(marker='o', markerfacecolor='red',
                                            markeredgecolor='red', markersize=8))

axes_t[1].set_title(f'LH cluster  ({ch_names[lh_peak]})', fontsize=9)
axes_t[2].set_title(f'RH cluster  ({ch_names[rh_peak]})', fontsize=9)

plt.tight_layout()
topo_path = os.path.join(OUT_DIR, 'cluster_topomaps.png')
fig_topo.savefig(topo_path, dpi=150)
plt.close(fig_topo)
print(f'Saved: {topo_path}')

# ── TOPOMAP: fixed clusters from make_ica_report.py ──────────────────────────
fig_topo2, axes_f = plt.subplots(1, 3, figsize=(13, 4))

grand_avg.plot_topomap(times=peak_time, axes=axes_f[0], show=False,
                       colorbar=False, time_unit='ms')
axes_f[0].set_title(f'Grand avg topomap\nat {peak_time*1000:.0f} ms', fontsize=9)

for ax, ch_list, label in [
        (axes_f[1], FIXED_LH, 'LH fixed cluster'),
        (axes_f[2], FIXED_RH, 'RH fixed cluster'),
]:
    mask_fixed = np.zeros((len(grand_avg.ch_names), len(grand_avg.times)), dtype=bool)
    for ch in ch_list:
        if ch in grand_avg.ch_names:
            mask_fixed[grand_avg.ch_names.index(ch), :] = True
    grand_avg.plot_topomap(times=peak_time, axes=ax, show=False, colorbar=False,
                           time_unit='ms', mask=mask_fixed,
                           mask_params=dict(marker='o', markerfacecolor='red',
                                            markeredgecolor='red', markersize=8))
    ax.set_title(f'{label}\n({ch_list[0]}…{ch_list[-1]})', fontsize=9)

plt.tight_layout()
topo2_path = os.path.join(OUT_DIR, 'cluster_topomaps_fixed.png')
fig_topo2.savefig(topo2_path, dpi=150)
plt.close(fig_topo2)
print(f'Saved: {topo2_path}')
        
# ── SUBJECT QC FIGURE ────────────────────────────────────────────────────────
# Per-subject grand average (all conditions) at each cluster — flag noisy subjects
# Noise metric: RMS of baseline period (-400 to 0 ms)
baseline_mask = times < 0

n_subs  = len(subjects)
n_times = len(times)

def cluster_mean(evoked, cluster_names):
    idx = [evoked.ch_names.index(c) for c in cluster_names if c in evoked.ch_names]
    return evoked.data[idx, :].mean(axis=0)

def fixed_cluster_mean(evoked, ch_list):
    available = [ch for ch in ch_list if ch in evoked.ch_names]
    if not available:
        return np.zeros(len(evoked.times))
    idx = [evoked.ch_names.index(ch) for ch in available]
    return evoked.data[idx, :].mean(axis=0)

# Compute per-subject grand average across all conditions
sub_grand = {}
for sub in subjects:
    evo_list = [all_evoked[sub][cid] for cid in ALL_CIDS]
    sub_grand[sub] = mne.grand_average(evo_list, drop_bads=False)

# RMS of baseline at RH cluster (most relevant for N170)
rms_vals = np.array([
    np.sqrt(np.mean(cluster_mean(sub_grand[sub], rh_cluster_names)[baseline_mask] ** 2))
    for sub in subjects
]) * 1e6   # µV

median_rms = np.median(rms_vals)
noisy_flag = rms_vals > 2.0 * median_rms

print(f'\nBaseline RMS at RH cluster (µV): median={median_rms:.2f}')
for sub, rms, flag in zip(subjects, rms_vals, noisy_flag):
    mark = ' *** NOISY' if flag else ''
    print(f'  {sub}: {rms:.2f} µV{mark}')

# Plot subject QC grid
n_cols = 6
n_rows = int(np.ceil(n_subs / n_cols))
fig_qc, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows), sharey=False)
axes = axes.flat

for ax, sub in zip(axes, subjects):
    rh_sig = cluster_mean(sub_grand[sub], rh_cluster_names) * 1e6
    lh_sig = cluster_mean(sub_grand[sub], lh_cluster_names) * 1e6
    ax.plot(times_ms, lh_sig, color='steelblue', lw=1, label='LH')
    ax.plot(times_ms, rh_sig, color='tomato',    lw=1, label='RH')
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4, ls='--')
    ax.set_xlim(-100, 500)
    ax.set_title(sub, fontsize=9,
                 color='red' if noisy_flag[list(subjects).index(sub)] else 'black')
    ax.tick_params(labelsize=7)

for ax in list(axes)[n_subs:]:
    ax.set_visible(False)

axes[0].legend(fontsize=7)
fig_qc.suptitle('Per-subject grand average — data-driven clusters  LH (blue) RH (red)\n'
                'Red subject labels = baseline RMS > 2× median', fontsize=10)
plt.tight_layout()
qc_path = os.path.join(OUT_DIR, 'subject_qc.png')
fig_qc.savefig(qc_path, dpi=130)
plt.close(fig_qc)
print(f'\nSaved QC figure: {qc_path}')

# QC figure for fixed clusters
fig_qc2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows), sharey=False)
axes2 = axes2.flat

for ax, sub in zip(axes2, subjects):
    rh_sig = fixed_cluster_mean(sub_grand[sub], FIXED_RH) * 1e6
    lh_sig = fixed_cluster_mean(sub_grand[sub], FIXED_LH) * 1e6
    ax.plot(times_ms, lh_sig, color='steelblue', lw=1, label='LH')
    ax.plot(times_ms, rh_sig, color='tomato',    lw=1, label='RH')
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4, ls='--')
    ax.set_xlim(-100, 500)
    ax.set_title(sub, fontsize=9,
                 color='red' if noisy_flag[list(subjects).index(sub)] else 'black')
    ax.tick_params(labelsize=7)

for ax in list(axes2)[n_subs:]:
    ax.set_visible(False)

axes2[0].legend(fontsize=7)
fig_qc2.suptitle('Per-subject grand average — fixed occipitotemporal clusters  LH (blue) RH (red)\n'
                 'Red subject labels = baseline RMS > 2× median (from data-driven RH)', fontsize=10)
plt.tight_layout()
qc2_path = os.path.join(OUT_DIR, 'subject_qc_fixed.png')
fig_qc2.savefig(qc2_path, dpi=130)
plt.close(fig_qc2)
print(f'Saved QC figure: {qc2_path}')

# ── EXTRACT CLUSTER SIGNALS ───────────────────────────────────────────────────
# signals[(hemi, cid)] → array (n_subjects, n_times)
# (cluster_mean and fixed_cluster_mean defined above)

# Data-driven clusters (paper method)
signals = {}
for hemi, cl_names in [('lh', lh_cluster_names), ('rh', rh_cluster_names)]:
    for cid in ALL_CIDS:
        mat = np.array([cluster_mean(all_evoked[sub][cid], cl_names)
                        for sub in subjects])
        signals[(hemi, cid)] = mat   # (n_subs, n_times)

# Fixed clusters from make_ica_report.py
signals_fixed = {}
for hemi, ch_list in [('lh', FIXED_LH), ('rh', FIXED_RH)]:
    for cid in ALL_CIDS:
        mat = np.array([fixed_cluster_mean(all_evoked[sub][cid], ch_list)
                        for sub in subjects])
        signals_fixed[(hemi, cid)] = mat

# ── BOOTSTRAP CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(mat, n_boot=N_BOOTSTRAP, alpha=0.05):
    idx   = np.arange(mat.shape[0])
    boots = np.array([mat[rng.choice(idx, size=len(idx), replace=True)].mean(axis=0)
                      for _ in range(n_boot)])
    return mat.mean(axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

# ── PLOT ─────────────────────────────────────────────────────────────────────
scale = 1e6   # V → µV

AWAY_COLOR   = '#e06b74'   # pink-red — away from observer (Ext-Int etc.)
TOWARD_COLOR = '#873838'   # dark maroon — toward observer (Ext-Dir etc.)

def plot_erp(ax, sig_dict, hemi, away_ids, toward_ids, title,
             away_label='Away', toward_label='Toward'):
    away_mat   = np.mean([sig_dict[(hemi, c)] for c in away_ids],   axis=0)
    toward_mat = np.mean([sig_dict[(hemi, c)] for c in toward_ids], axis=0)

    mn_a, lo_a, hi_a = bootstrap_ci(away_mat)
    mn_t, lo_t, hi_t = bootstrap_ci(toward_mat)

    mn_a *= scale;  lo_a *= scale;  hi_a *= scale
    mn_t *= scale;  lo_t *= scale;  hi_t *= scale

    ax.fill_between(times_ms, lo_a, hi_a, color=AWAY_COLOR,   alpha=0.25)
    ax.fill_between(times_ms, lo_t, hi_t, color=TOWARD_COLOR, alpha=0.25)
    ax.plot(times_ms, mn_a, color=AWAY_COLOR,   lw=2, label=away_label)
    ax.plot(times_ms, mn_t, color=TOWARD_COLOR, lw=2, label=toward_label)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.axvspan(CLUSTER_TMIN * 1000, CLUSTER_TMAX * 1000, alpha=0.08, color='gray')
    ax.set_xlim(-100, 500)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='lower right')

# Data-driven cluster figures (paper method)
for figkey, cfg in COND_GROUPS.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f'Non-social task — {cfg["title"]}  (n={n_subs})\n'
                 f'Data-driven cluster (paper method)', fontsize=10)
    plot_erp(axes[0], signals, 'lh', cfg['away'], cfg['toward'],
             f'LH  ({ch_names[lh_peak]})',
             away_label=cfg['away_label'], toward_label=cfg['toward_label'])
    plot_erp(axes[1], signals, 'rh', cfg['away'], cfg['toward'],
             f'RH  ({ch_names[rh_peak]})',
             away_label=cfg['away_label'], toward_label=cfg['toward_label'])
    axes[1].get_legend().remove()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'{figkey}_erp.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')

# Fixed cluster figures (from make_ica_report.py)
for figkey, cfg in COND_GROUPS.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f'Non-social task — {cfg["title"]}  (n={n_subs})\n'
                 f'Fixed occipitotemporal cluster (make_ica_report.py)', fontsize=10)
    plot_erp(axes[0], signals_fixed, 'lh', cfg['away'], cfg['toward'],
             f'LH fixed  ({", ".join(FIXED_LH[:3])}...)',
             away_label=cfg['away_label'], toward_label=cfg['toward_label'])
    plot_erp(axes[1], signals_fixed, 'rh', cfg['away'], cfg['toward'],
             f'RH fixed  ({", ".join(FIXED_RH[:3])}...)',
             away_label=cfg['away_label'], toward_label=cfg['toward_label'])
    axes[1].get_legend().remove()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'{figkey}_erp_fixed.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')

print('\nDone.')
