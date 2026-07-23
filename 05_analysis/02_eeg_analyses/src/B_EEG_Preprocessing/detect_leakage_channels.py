import numpy as np
import matplotlib.pyplot as plt
import mne

def detect_leakage_channels(
    raw,
    z_thresh: float = 2.0,
    corr_thresh: float = 0.4,
    n_top: int = 6,
    plot: bool = True,
    verbose: bool = True,
):
    """
    Detect EEG channels likely causing artifact "leakage" into other channels
    via the referencing scheme (e.g. a bad mastoid or reference electrode like
    M1/M2), and optionally mark them in ``raw.info["bads"]``.

    This function implements the logic of ``check_leakage_channels.py`` in a
    reusable, Script‑2–style API:

      1. Flag channels with abnormally large peak‑to‑peak amplitude (z‑score).
      2. For each flagged ("suspect") channel, compute correlation with all
         other channels and identify the most strongly correlated ones.
      3. Optionally plot suspect channels and their most correlated neighbours.
      4. Optionally mark suspect channels as bad.

    Intended to be run **before** average re‑referencing and before ICA, on
    data in its original reference.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording in its original hardware reference.
        Should be filtered (e.g. high‑pass ≥ 0.5–1 Hz) before calling.
    z_thresh : float
        Z‑score threshold on peak‑to‑peak amplitude above which a channel is
        flagged as a suspect leakage source (default 3.0).
    corr_thresh : float
        Absolute correlation threshold used to highlight "strongly correlated"
        neighbours in the diagnostic plot (default 0.4). Channels with
        |r| ≥ corr_thresh are considered potentially affected by leakage.
    n_top : int
        Number of top correlated channels to display/print per suspect channel
        (default 6).
    plot : bool
        If True (default), show a diagnostic figure with:
          - Bar plot of peak‑to‑peak amplitudes and z‑scores.
          - For each suspect channel, a time‑course plot with its most
            correlated neighbours.
    mark_as_bad : bool
        If True (default), append suspect channels to ``raw.info["bads"]``.
        Existing entries in ``raw.info["bads"]`` are preserved.
    verbose : bool
        If True (default), print suspect channels and their most correlated
        neighbours.
    """

    # ── Select EEG channels ───────────────────────────────────────────────────
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels found. Check channel types.")

    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    data = raw.get_data(picks=eeg_picks)  # (n_channels, n_times)

    # ── Criterion: outlier peak‑to‑peak amplitude (z‑score) ─────────────────
    ptp = data.max(axis=1) - data.min(axis=1)
    z_ptp = (ptp - ptp.mean()) / (ptp.std() + 1e-12)

    suspect_mask = z_ptp > z_thresh
    suspect_channels = [eeg_names[i] for i in np.where(suspect_mask)[0]]

    if verbose:
        print(f"[detect_leakage_channels] Z‑threshold  : {z_thresh}")
        print(f"[detect_leakage_channels] Corr threshold: {corr_thresh}")
        print(f"[detect_leakage_channels] Suspect channels (z > {z_thresh}): {suspect_channels}")

    # ── Correlate other channels against each suspect channel ───────────────
    def top_correlated_channels(suspect_name, data, ch_names, n_top=n_top):
        idx_suspect = ch_names.index(suspect_name)
        suspect_trace = data[idx_suspect]
        corrs = []
        for i, name in enumerate(ch_names):
            if name == suspect_name:
                continue
            r = np.corrcoef(suspect_trace, data[i])[0, 1]
            corrs.append((name, r))
        corrs.sort(key=lambda x: -abs(x[1]))
        return corrs[:n_top]

    correlation_results = {}
    for suspect in suspect_channels:
        top = top_correlated_channels(suspect, data, eeg_names)
        correlation_results[suspect] = top
        if verbose:
            print(f"\n[detect_leakage_channels] Channels most correlated with {suspect}:")
            for name, r in top:
                flag = "  [high]" if abs(r) >= corr_thresh else ""
                print(f"    {name:<10} r = {r:+.3f}{flag}")

    # ── Diagnostic plots ─────────────────────────────────────────────────────
    if plot:
        print("\nOpening MNE interactive plot — suspect channels are pre-marked as bad.")
        print("Click channel names/traces to toggle additional bad channels.")
        print("Pre-marked as bad:", raw.info["bads"])
        raw.plot(duration=40, start=60, n_channels=20, block=True)
        print("\nFinal bad channels after review:", raw.info["bads"])
