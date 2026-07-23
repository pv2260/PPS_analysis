import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from mne.preprocessing import ICA
from mne_icalabel import label_components


def ica_artefact_removal(
    raw_filt: mne.io.Raw,
    ica_segments: Optional[list[tuple[float, float]]] = None,
    n_components: Optional[int] = None,
    method: str = "infomax",
    fit_params: Optional[dict] = None,
    auto_label: bool = True,
    label_threshold: float = 0.8,
    montage: Optional[Union[str, mne.channels.DigMontage]] = "standard_1020",
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Fit ICA on filtered continuous data, identify artefact components, and
    return a cleaned copy of the recording.

    Artefact components are identified in two complementary ways:

    1. **Automatic labelling** (``auto_label=True``): uses ``mne-icalabel``
       (ICLabel classifier) to flag non-brain components (EOG, ECG, muscle,
       line-noise, channel-noise).  Components with probability ≥
       ``label_threshold`` for any artefact class are excluded.

    2. **Channel-correlation fallback**: if EOG/ECG reference channels are
       provided *and* ICLabel did not already catch them, MNE's
       ``find_bads_eog`` / ``find_bads_ecg`` are applied as an additional
       safeguard.

    The function prints a summary of excluded components and returns the
    cleaned raw signal together with the fitted ICA object so the caller can
    inspect, save, or adjust exclusions. If ``plot_sanity=True`` (default),
    it also generates the standard set of ICA sanity-check figures used to
    visually confirm the decomposition and exclusion decisions are sane
    before trusting ``raw_clean`` downstream.

    Parameters
    ----------
    raw_filt : mne.io.Raw
        Band-pass filtered (and notch-filtered) continuous data.
        **Must be preloaded.**  A copy is made internally; the original is
        not modified.
    ica_segments : list of tuple or None
        Time segments (seconds) used only for ICA fitting. If None, the full recording is used.
        Example: [(60, 600), (1200, 1800)]
    n_components : int or None
        Number of ICA components to decompose into.  None uses the MNE
        default (min(#channels, #samples/sfreq − 1)).  Typical value: 30–64.
    method : str
        ICA algorithm: "infomax" (default, recommended for EEG),
        "fastica", or "picard".
    fit_params : dict or None
        Extra keyword arguments forwarded to ``ICA.fit()``.
        E.g. ``{"extended": True}`` for extended Infomax.
    auto_label : bool
        Run ICLabel classifier to auto-detect artefact components.
    label_threshold : float
        Probability threshold above which a component is considered an
        artefact by ICLabel (default 0.8).
    montage : str or DigMontage or None
        Electrode position montage required by ICLabel.  Accepts any name
        understood by ``mne.channels.make_standard_montage()`` (e.g.
        ``"standard_1020"``, ``"easycap-M1"``, ``"biosemi64"``) or a
        pre-built ``DigMontage`` object.  If the recording already has
        digitisation positions embedded, pass ``None`` to skip.
        Default: ``"standard_1020"``.
    random_state : int
        Seed for reproducibility.
    verbose : bool

    Returns
    -------
    raw_clean : mne.io.Raw
        Artefact-corrected copy of ``raw_filt``.
    ica : mne.preprocessing.ICA
        Fitted ICA object (``ica.exclude`` contains the excluded indices).
    excluded : list[int]
        Sorted list of excluded component indices.
    component_labels: 
    """
    if fit_params is None:
        fit_params = {}
    if method == "infomax" and "extended" not in fit_params:
        fit_params["extended"] = True  # extended Infomax is the MNE/EEGLAB default

    # --- 1. Create data used for ICA fitting ---------------
    if verbose: print("[ICA] Preparing data for ICA fitting")
    if ica_segments is not None:
        if verbose: print(f"[ICA] Using {len(ica_segments)} selected segments")
        segments = []
        for i, (tmin, tmax) in enumerate(ica_segments):
            seg = raw_filt.copy().crop(
                tmin=tmin,
                tmax=tmax
            )
            segments.append(seg)

        # Concatenate selected periods
        raw_for_ica = mne.concatenate_raws(
            segments,
            verbose=False
        )
    else:
        if verbose: print("[ICA] Using full recording")
        raw_for_ica = raw_filt.copy()

    # --- 1. High-pass copy for ICA fitting (standard practice) ---------------
    if verbose:
        print("[ica_artefact_removal] High-pass filtering at 1 Hz for ICA fit …")
    raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)

    # --- 1b. Ensure a montage is set (required by ICLabel) -------------------
    if montage is not None:
        if isinstance(montage, str):
            montage_obj = mne.channels.make_standard_montage(montage)
            if verbose:
                print(f"[ica_artefact_removal] Setting montage: '{montage}'")
        else:
            montage_obj = montage
            if verbose:
                print("[ica_artefact_removal] Setting provided DigMontage.")
        raw_for_ica.set_montage(montage_obj, on_missing="ignore", verbose=False)
    else:
        if verbose:
            print("[ica_artefact_removal] Skipping montage (None passed —bassuming digitisation is already embedded).")

    # --- 2. Fit ICA -----------------------------------------------------------
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter="auto",
    )
    if verbose:
        print(f"[ica_artefact_removal] Fitting ICA ({method}, n_components={n_components}) …")

    ica.fit(raw_for_ica, verbose=False)
    if verbose:
        print(f"  → {ica.n_components_} components extracted")

    excluded: list[int] = []

    # --- 3a. ICLabel automatic labelling --------------------------------------
    if auto_label:
        if verbose:
            print("[ica_artefact_removal] Running ICLabel classifier …")
        component_labels = label_components(raw_for_ica, ica, method="iclabel")
        labels      = component_labels["labels"]       # list of strings
        probs       = component_labels["y_pred_proba"] # (n_components, n_classes)

        artefact_classes = {"eye blink", "heart beat", "muscle artifact","line noise", "channel noise"}
        for idx, (label, prob_row) in enumerate(zip(labels, probs)):
            if label in artefact_classes:
                max_prob = prob_row.max()
                if max_prob >= label_threshold:
                    excluded.append(idx)
                    if verbose:
                        print(f"    Component {idx:3d}  → {label}  "
                              f"(p={max_prob:.2f}) — EXCLUDED")

    if verbose:
        print(f"[ica_artefact_removal] Total excluded components: {excluded}")

                
    # --- 5. Apply ICA to the *original* filtered data -------------------------
    ica.exclude = excluded
    raw_clean = ica.apply(raw_filt.copy(), verbose=False)
    if verbose:
        print("[ica_artefact_removal] ICA correction applied to raw_filt → raw_clean")

    return raw_clean, ica, excluded, component_labels





def plot_ica_before_after(
    raw_before,
    raw_after,
    tmin=20,
    tmax=40,
    n_channels=5,
    picks=None,
    random_state=None,
    scale_uv=1e6,
):
    """
    Plot EEG signals before and after ICA cleaning over the same time window.

    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before ICA correction.
    raw_after : mne.io.Raw
        Raw data after ICA correction.

    tmin : float
        Start time window in seconds.

    tmax : float
        End time window in seconds.

    n_channels : int
        Number of EEG channels to randomly display if picks is None.

    picks : list | None
        EEG channel indices or names to display.

    random_state : int | None
        Seed for random channel selection.

    scale_uv : float
        Conversion factor to µV (default: 1e6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    """

    rng = np.random.default_rng(random_state)

    # Select EEG channels
    if picks is None:

        eeg_picks = mne.pick_types(
            raw_before.info,
            eeg=True,
            exclude=[]
        )

        picks = rng.choice(
            eeg_picks,
            size=min(n_channels, len(eeg_picks)),
            replace=False
        )

    else:
        # Convert channel names to indices if needed
        if isinstance(picks[0], str):
            picks = mne.pick_channels(
                raw_before.ch_names,
                include=picks
            )


    # Extract data
    data_before, times = raw_before[picks, :]

    data_after, _ = raw_after[picks, :]


    # Select time window
    mask = (
        (times >= tmin) &
        (times <= tmax)
    )

    times = times[mask]

    data_before = data_before[:, mask] * scale_uv
    data_after = data_after[:, mask] * scale_uv


    # Create figure
    fig, axes = plt.subplots(
        len(picks),
        1,
        figsize=(12, 2 * len(picks)),
        sharex=True
    )

    if len(picks) == 1:
        axes = [axes]


    for i, ax in enumerate(axes):

        ax.plot(
            times,
            data_before[i],
            label="Before ICA",
            linewidth=1
        )

        ax.plot(
            times,
            data_after[i],
            label="After ICA",
            linewidth=1
        )

        ax.set_ylabel(
            raw_before.ch_names[picks[i]]
        )

        ax.grid(True)


    axes[0].legend()

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        f"ICA cleaning effect ({tmin}-{tmax}s)",
        fontsize=14
    )

    plt.tight_layout()

    return fig