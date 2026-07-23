def get_segments_to_keep(task, subjectID):
    """
    Return the list of (start, stop) segments to keep for a given task and subject.

    Parameters
    ----------
    task : str e.g. "Task1_PPS" or "Task2_HitOrMiss"
    subjectID : str e.g. "Pilote001"

    Returns
    -------
    list of tuple
        [(start1, stop1), (start2, stop2), ...]
    """

    segments = {
        "Task1_PPS": {
            "Pilote001": [(50, 620), (660,1200), (1300,1750)],
            "Pilote002": [(50, 1350), (1500,2100)],
        },

        "Task2_HitOrMiss": {
            "Pilote001": [(40, 1550)],
            "Pilote002": [(110, 1155)],
        },
    }

    try:
        return segments[task][subjectID]
    except KeyError:
        raise ValueError(f"No segment defined for task='{task}', subject='{subjectID}'")