import numpy as np


def average_grouped_tfr(
    tfr_single_dict,
    groups,
    verbose=True
):
    """
    Group single-trial TFRs across conditions.

    Parameters
    ----------
    tfr_single_dict : dict
        Dictionary of EpochsTFR objects.

    groups : dict
        Mapping from grouped condition names to lists of conditions.

    verbose : bool
        Print progress.

    Returns
    -------
    tfr_grouped_single : dict
        Dictionary of grouped EpochsTFR objects.

    tfr_grouped_average : dict
        Dictionary of grouped AverageTFR objects.
    """

    tfr_grouped_single = {}
    tfr_grouped_average = {}

    for group_name, conditions in groups.items():

        if verbose:
            print(f"\nProcessing {group_name}")

        tfr_list = []

        for condition in conditions:

            if condition in tfr_single_dict:

                tfr_list.append(tfr_single_dict[condition])

                if verbose:
                    print(
                        f"  {condition}: "
                        f"{len(tfr_single_dict[condition])} epochs"
                    )

            elif verbose:
                print(f"  Missing: {condition}")

        if len(tfr_list) == 0:
            continue

        # --------------------------------------------------
        # Concatenate epochs
        # --------------------------------------------------

        tfr_concat = tfr_list[0].copy()

        tfr_concat._data = np.concatenate(
            [tfr.data for tfr in tfr_list],
            axis=0
        )

        # Concatenate events if present
        if hasattr(tfr_concat, "events"):
            tfr_concat.events = np.concatenate(
                [tfr.events for tfr in tfr_list],
                axis=0
            )

        # Concatenate metadata if present
        if getattr(tfr_concat, "metadata", None) is not None:
            import pandas as pd

            metadata = [
                tfr.metadata
                for tfr in tfr_list
                if tfr.metadata is not None
            ]

            if len(metadata):
                tfr_concat.metadata = pd.concat(
                    metadata,
                    ignore_index=True
                )

        # Update nave
        tfr_concat._nave = tfr_concat.data.shape[0]

        # Store grouped single-trial TFR
        tfr_grouped_single[group_name] = tfr_concat

        # Store grouped average
        tfr_grouped_average[group_name] = tfr_concat.average()

        if verbose:
            print(f"  Total epochs: {len(tfr_concat)}")

    return tfr_grouped_single, tfr_grouped_average


# import numpy as np

# def average_grouped_tfr(
#     tfr_single_dict,
#     groups,
#     average_bool=True,
#     verbose=True
# ):
#     """
#     Concatenate TFR epochs across conditions belonging to the same group
#     and optionally average them.

#     Parameters
#     ----------
#     tfr_single_dict : dict
#         Dictionary containing EpochsTFR objects for each condition.

#     groups : dict
#         Dictionary mapping group names to lists of conditions.

#     average_bool : bool, default=True
#         If True, return averaged TFR (AverageTFR).
#         If False, return concatenated epoched TFR (EpochsTFR).

#     verbose : bool, default=True
#         Print processing information.

#     Returns
#     -------
#     tfr_grouped_dict : dict
#         Dictionary containing grouped TFR objects.
#         Values are AverageTFR if average_bool=True,
#         otherwise EpochsTFR.
#     """

#     tfr_grouped_dict = {}

#     for group_name, conditions in groups.items():

#         if verbose:
#             print(f"\nProcessing {group_name}")

#         tfr_list = []

#         for condition in conditions:

#             if condition in tfr_single_dict:
#                 tfr_list.append(
#                     tfr_single_dict[condition]
#                 )

#                 if verbose:
#                     print(
#                         f"  {condition}: "
#                         f"{len(tfr_single_dict[condition])} epochs"
#                     )

#             else:
#                 if verbose:
#                     print(f"  Missing: {condition}")


#         if len(tfr_list) == 0:
#             continue


#         # --------------------------------------------------
#         # Concatenate trials manually
#         # --------------------------------------------------

#         tfr_concat = tfr_list[0].copy()

#         tfr_concat._data = np.concatenate(
#             [
#                 tfr.data
#                 for tfr in tfr_list
#             ],
#             axis=0
#         )

#         # Update number of epochs
#         tfr_concat._nave = tfr_concat.data.shape[0]


#         # --------------------------------------------------
#         # Average or keep epochs
#         # --------------------------------------------------

#         if average_bool:
#             tfr_grouped_dict[group_name] = tfr_concat.average()

#             if verbose:
#                 print(
#                     f"  -> Averaged epochs: {tfr_concat.data.shape[0]}"
#                 )

#         else:
#             tfr_grouped_dict[group_name] = tfr_concat

#             if verbose:
#                 print(
#                     f"  -> Returned epoched TFR: {tfr_concat.data.shape[0]} epochs"
#                 )


#     return tfr_grouped_dict

# # import numpy as np

# # def average_grouped_tfr(
# #     tfr_single_dict,
# #     groups,
# #     verbose=True
# # ):

# #     tfr_average_dict = {}

# #     for group_name, conditions in groups.items():

# #         if verbose:
# #             print(f"\nProcessing {group_name}")

# #         tfr_list = []

# #         for condition in conditions:

# #             if condition in tfr_single_dict:
# #                 tfr_list.append(
# #                     tfr_single_dict[condition]
# #                 )

# #                 if verbose:
# #                     print(
# #                         f"  {condition}: "
# #                         f"{len(tfr_single_dict[condition])} epochs"
# #                     )

# #             else:
# #                 print(f"  Missing: {condition}")


# #         if len(tfr_list) == 0:
# #             continue


# #         # --------------------------------------------------
# #         # Concatenate trials manually
# #         # --------------------------------------------------

# #         tfr_concat = tfr_list[0].copy()

# #         tfr_concat._data = np.concatenate(
# #             [
# #                 tfr.data
# #                 for tfr in tfr_list
# #             ],
# #             axis=0
# #         )


# #         # Update number of epochs
# #         tfr_concat._nave = tfr_concat.data.shape[0]


# #         # --------------------------------------------------
# #         # Average across all trials
# #         # --------------------------------------------------

# #         tfr_average_dict[group_name] = (
# #             tfr_concat.average()
# #         )


# #         if verbose:
# #             print(
# #                 f"  -> Total epochs: {tfr_concat.data.shape[0]}"
# #             )


# #     return tfr_average_dict