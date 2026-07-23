import os
import numpy as np
import matplotlib
import re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from matplotlib.colors import TwoSlopeNorm

def plot_spectrograms_on_capmap(
    tfr,
    subjectID,
    task,
    trial_type,
    n_epochs,
    data_path,
    interpolation_config=None,
    groups=None,
    t_min=None,
    t_max=None,
    f_min=None,
    f_max=None,
    scale_db=10,
    fig_height=17,
    canvas_aspect=1.5,
    panel_inch=1.2,
    panel_aspect=1.4,
    f_front=1.2,
    f_parietal=1.6,
    f_po=1.8,
    f_other=1.0,
    center_gain=0.5,
    margin=0.05,
    cmap="RdBu_r",
    colormin=None,
    colormax=None
):

    tfr = tfr.copy()

    if t_min is not None or t_max is not None:
        tfr.crop(
            tmin=t_min,
            tmax=t_max,
        )

    if f_min is not None or f_max is not None:
        tfr.crop(
            fmin=f_min,
            fmax=f_max,
        )    

    data = tfr.data.copy()

    # # -----------------------------
    # # Baseline correction
    # # -----------------------------
    # if baseline is not None:
    #     data = rescale(
    #         data,
    #         tfr.times,
    #         baseline,
    #         mode,
    #         copy=False
    #     )


    # -----------------------------
    # Scale dB
    # -----------------------------
    data *= scale_db


    # -----------------------------
    # Global color scale
    # -----------------------------
    if colormin is None or colormax is None:

        # Flatten all channels, frequencies, and times
        all_values = data.flatten()

        # Robust quantiles
        q_low = np.nanquantile(all_values, 0.05)
        q_high = np.nanquantile(all_values, 0.995)

        if colormin is None:
            colormin = q_low

        if colormax is None:
            if q_high <= 0:
                colormax = 0.5
                print(
                    f"Warning: q_high ({q_high:.3f}) <= 0 Setting colormax to 0.5."
                )
            else:
                colormax = q_high


    # -----------------------------
    # Colormap normalization
    # -----------------------------
    norm = TwoSlopeNorm(
        vmin=colormin,
        vcenter=0,
        vmax=colormax
)



    # -----------------------------
    # Vibration onset
    # -----------------------------
    # vibration_time = None

    if (
        interpolation_config is not None
    ):

        interp_key = get_interpolation_key(
            trial_type,
            groups
        )


    # -----------------------------
    # Layout
    # -----------------------------
    layout = mne.channels.find_layout(
        tfr.info
    )

    names = layout.names
    xy = layout.pos[:, :2].copy()


    xy -= xy.min(axis=0)
    xy /= np.maximum(
        xy.max(axis=0),
        1e-12
    )


    # x expansion
    xc = 0.5

    dx = xy[:, 0] - xc

    d = np.abs(dx)
    d /= d.max() + 1e-12

    spread = (
        1
        + center_gain * d**2
    )


    spread *= np.array(
        [
            region_base_factor(
                ch,
                f_front,
                f_parietal,
                f_po,
                f_other
            )
            for ch in names
        ]
    )


    xy[:, 0] = xc + dx * spread

    xy = np.clip(
        xy,
        0.02,
        0.98
    )

    xy -= xy.min(axis=0)
    xy /= np.maximum(
        xy.max(axis=0),
        1e-12
    )

    xy = (
        xy * (1 - 2 * margin)
        + margin
    )



    # -----------------------------
    # Figure
    # -----------------------------
    fw = fig_height * canvas_aspect

    fig = plt.figure(
        figsize=(fw, fig_height),
        facecolor="white"
    )


    pw = (
        panel_inch
        * panel_aspect
        / fw
    )

    ph = (
        panel_inch
        / fig_height
    )


    im = None



    # -----------------------------
    # Plot channels
    # -----------------------------
    for i, ch in enumerate(names):

        if ch not in tfr.ch_names:
            continue


        idx = tfr.ch_names.index(ch)

        x, y = xy[i]


        ax = fig.add_axes(
            [
                np.clip(x-pw/2, 0, 1-pw),
                np.clip(y-ph/2, 0, 1-ph),
                pw,
                ph
            ]
        )


        im = ax.imshow(
            data[idx],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=[
                tfr.times[0],
                tfr.times[-1],
                tfr.freqs[0],
                tfr.freqs[-1],
            ],
        )

        # Frequency references
        ax.axhline(
            4,
            color="k",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            8,
            color="k",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            13,
            color="k",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            30,
            color="k",
            linestyle="--",
            linewidth=1
        )

        if f_min is not None or f_max is not None:
            ax.set_ylim(
                f_min,
                f_max
            )


        ax.set_title(
            ch,
            fontsize=7
        )

        # ax.set_xticks([])
        # ax.set_yticks([])
        # X-axis: time in seconds
        ax.set_xticks(
            np.linspace(
                tfr.times[0],
                tfr.times[-1],
                5
            )
        )

        ax.set_xlabel(
            "Time (s)",
            fontsize=6
        )

        ax.tick_params(
            axis="x",
            labelsize=6
        )


        # Y-axis: frequency in Hz
        ax.set_yticks(
            np.linspace(
                tfr.freqs[0],
                tfr.freqs[-1],
                4
            )
        )

        ax.set_ylabel(
            "Hz",
            fontsize=6
        )

        ax.tick_params(
            axis="y",
            labelsize=6
        )


        # Stimulus onset
        # FIX: axvline needs a TIME value (matching the imshow `extent`),
        # not a sample index. `np.where(tfr.times == 0)[0][0]` returns an
        # index (e.g. 40), which — on an axis whose extent is in seconds —
        # forces matplotlib to autoscale the x-axis out to that index value,
        # squashing the actual spectrogram into a thin sliver at the left
        # edge (this is why the panels looked "empty").
        ax.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=1
        )


        # Vibration onset
        # FIX: same issue — use the time value directly instead of
        # converting it back to an index with np.argmin.
        # if vibration_time is not None:
        if interpolation_config != None and interp_key in interpolation_config:
            cfg = interpolation_config[interp_key]
            if cfg["type"] == "vibrotactile":
                ax.axvline(
                    cfg["N_sec_before_vibration"],
                    color="red",
                    linestyle="--",
                    linewidth=1
                )




    # -----------------------------
    # Colorbar
    # -----------------------------
    cax = fig.add_axes(
        [
            0.82,
            0.04,
            0.12,
            0.015
        ]
    )

    fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal"
    )



    fig.suptitle(
        f"{subjectID} - {task} (n={n_epochs})",
        fontsize=14
    )


    # -----------------------------
    # Save
    # -----------------------------
    save_dir = os.path.join(
        data_path,
        subjectID,
        task,
        "Output",
        "CapMap"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    fig.savefig(
        os.path.join(
            save_dir,
            f"{subjectID}_{task}_{trial_type}_{n_epochs}.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close(fig)

    return fig



def region_base_factor(
    ch_name,
    f_front=1.0,
    f_parietal=1.15,
    f_po=1.3,
    f_other=1.05
):
    name = ch_name.upper()

    if name.startswith("FP") or name.startswith("F"):
        return f_front
    elif name.startswith("PO"):
        return f_po
    elif name.startswith("P"):
        return f_parietal
    else:
        return f_other




def get_interpolation_key(event_name, groups=None):
    """
    Extract interpolation key from original or grouped event names.

    Parameters
    ----------
    event_name : str
        Condition or grouped condition name.

    groups : dict | None
        Dictionary defining grouped conditions.
        Example:
        {
            "Both_Near_Fast": [
                "Both_D1_Narrow_Fast",
                "Both_D2_Narrow_Fast"
            ]
        }

    Returns
    -------
    str | None
        Key matching interpolation_config.
    """

    # --------------------------------------------------
    # If grouped name, replace by first original condition
    # --------------------------------------------------
    if groups is not None:
        event_name = event_name.removesuffix("-tfr.h5")
        if event_name in groups:

            # Take first condition as representative
            # event_name = groups[event_name][0]
            return event_name


    # --------------------------------------------------
    # Extract D and speed
    # --------------------------------------------------
    match = re.search(
        r"(D\d+_Narrow_(?:Fast|Slow))",
        event_name
    )

    if match:
        return match.group(1)


    # --------------------------------------------------
    # Visual conditions
    # --------------------------------------------------
    if "VisualOnly" in event_name:
        return event_name


    return None