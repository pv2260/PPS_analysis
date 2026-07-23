import mne
from typing import Dict, List

def apply_manual_bad_channels(
    raw: mne.io.Raw,
    participant_id: str,
    interpolate: bool = False,
    verbose: bool = True,
):
    """
    Manually mark bad channels for a given participant using a predefined
    dictionary (stored inside this function), then (optionally) interpolate
    them and return a cleaned Raw object.

    This function is useful when you have a priori knowledge of bad channels
    (e.g., from visual inspection, Script 1/2 outputs, or reproducibility across
    sessions) and want to apply them programmatically in a batch pipeline.

    The function does NOT modify the input `raw` in-place if `interpolate=True`;
    it returns a copy with bad channels interpolated. If `interpolate=False`,
    it modifies `raw.info["bads"]` in-place and returns the same object.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording. Must have a montage set if `interpolate=True`.
    participant_id : str
        Identifier for the participant (e.g., "P01", "subject_003"). Must be a key
        in the internal bad_channels_dict.
    interpolate : bool
        If True (default False), interpolate the bad channels after marking them.
        This creates a copy of the Raw object, interpolates the bads, and returns
        the cleaned version. The original `raw` is not modified in this case.
        If False, the function only updates `raw.info["bads"]` in-place and returns
        the same object.
    verbose : bool
        If True (default), print which channels are being marked as bad and whether
        interpolation was performed.

    Returns
    -------
    raw_clean : mne.io.Raw
        If `interpolate=True`: a new Raw object with bad channels interpolated.
        If `interpolate=False`: the same Raw object with `raw.info["bads"]` updated.
    """
    # === Manually filled dictionary: participant_id -> list of bad channel names ===
    bad_channels_dict: Dict[str, List[str]] = {
        "PamTest": [],
        "Pilote001": ["M1", "M2", "FC1", "FC2"],
        "Pilote002": ["M1", "M2", 'Fp1', 'FC1', 'M1', 'P7', 'F5', 'AF7'],
    }
    # ================================================================================

    if participant_id not in bad_channels_dict:
        raise KeyError(
            f"Participant ID '{participant_id}' not found in bad_channels_dict. "
            f"Available IDs: {list(bad_channels_dict.keys())}"
        )

    bad_channels = bad_channels_dict[participant_id]

    if verbose:
        print(f"[apply_manual_bad_channels] Participant: {participant_id}")
        print(f"[apply_manual_bad_channels] Marking {len(bad_channels)} channel(s) as bad: {bad_channels}")

    # Create a copy to avoid modifying the original if we plan to interpolate
    if interpolate:
        raw_copy = raw.copy()
        raw_copy.info["bads"] = list(set(raw_copy.info.get("bads", [])) | set(bad_channels))
    else:
        raw_copy = raw
        raw_copy.info["bads"] = list(set(raw_copy.info.get("bads", [])) | set(bad_channels))

    if interpolate:
        if raw_copy.info.get_montage() is None:
            raise ValueError(
                "No montage set on the Raw object. Cannot interpolate bad channels. "
                "Run `raw.set_montage(...)` first or set `interpolate=False`."
            )
        if verbose:
            print(f"[apply_manual_bad_channels] Interpolating {len(bad_channels)} bad channel(s)...")
        raw_copy.interpolate_bads(reset_bads=True)
        if verbose:
            print(f"[apply_manual_bad_channels] Interpolation complete. Bad channels reset.")

    return raw_copy