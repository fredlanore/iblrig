import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.core.dtypes.concat import union_categoricals

log = logging.getLogger(__name__)
RE_PATTERN_EVENT = re.compile(r'^(\D+\d)_?(.+)$')


def load_task_jsonable(jsonable_file: str | Path, offset: int | None = None) -> tuple[pd.DataFrame, list[Any]]:
    """
    Reads in a task data jsonable file and returns a trials dataframe and a bpod data list.

    Parameters
    ----------
    - jsonable_file (str): full path to jsonable file.
    - offset (int or None): The offset to start reading from (default: None).

    Returns
    -------
    - tuple: A tuple containing:
        - trials_table (pandas.DataFrame): A DataFrame with the trial info in the same format as the Session trials table.
        - bpod_data (list): timing data for each trial
    """
    trials_table = []
    with open(jsonable_file) as f:
        if offset is not None:
            f.seek(offset, 0)
        for line in f:
            trials_table.append(json.loads(line))

    # pop-out the bpod data from the table
    bpod_data = []
    for td in trials_table:
        bpod_data.append(td.pop('behavior_data'))

    trials_table = pd.DataFrame(trials_table)
    return trials_table, bpod_data


def bpod_session_events_to_dataframe(bpod_data: list[dict[str, Any]], trials: int | list[int] | slice | None = None):
    """
    Convert Bpod session data into a single Pandas DataFrame.

    Parameters
    ----------
    bpod_data : list of dict
        A list of dictionaries as returned by load_task_jsonable, where each dictionary contains data for a single trial.
    trials : int, list of int, slice, or None, optional
        Specifies which trials to include in the DataFrame. All trials are included by default.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing combined event data from the specified trials, with columns for time, event, channel,
        and value.
    """
    # define trial index
    if trials is None:
        trials = range(len(bpod_data))
    elif isinstance(trials, int):
        return _bpod_trial_events_to_dataframe(bpod_data[trials], trials)
    elif isinstance(trials, slice):
        trials = range(len(bpod_data))[trials]

    # loop over requested trials
    dataframes = []
    for trial in trials:
        dataframes.append(_bpod_trial_events_to_dataframe(bpod_data[trial], trial))

    # combine trials into a single dataframe
    categories_event = union_categoricals([df['Event'] for df in dataframes])
    categories_channel = union_categoricals([df['Channel'] for df in dataframes])
    for df in dataframes:
        df['Event'] = df['Event'].cat.set_categories(categories_event.categories)
        df['Channel'] = df['Channel'].cat.set_categories(categories_channel.categories)
    return pd.concat(dataframes, ignore_index=True)


def _bpod_trial_events_to_dataframe(bpod_trial_data: dict[str, Any], trial: int) -> pd.DataFrame:
    """
    Convert a single Bpod trial's data into a Pandas DataFrame.

    Parameters
    ----------
    bpod_trial_data : dict
        A dictionary containing data for a single trial, including timestamps and events.
    trial : int
        An integer representing the trial index.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the event data for the specified trial, with columns for time, event, channel, and value.
    """
    trial_start = bpod_trial_data['Trial start timestamp']
    trial_end = bpod_trial_data['Trial end timestamp']

    # convert bpod trial data to list of tuples, containing timestamps and event-names
    event_list = [(time, event) for event, times in bpod_trial_data['Events timestamps'].items() for time in times]
    event_list = [(0, 'TrialStart')] + sorted(event_list) + [(trial_end - trial_start, 'TrialEnd')]

    # create dataframe
    df = pd.DataFrame(data=event_list, columns=['Time', 'Event'])
    df['Time'] = pd.to_timedelta(df['Time'] + trial_start, unit='seconds')
    df['Event'] = df['Event'].astype('category')
    df.insert(1, 'Trial', pd.to_numeric(pd.Series(trial, index=df.index), downcast='unsigned'))

    # deduce channels and values from event names
    df[['Channel', 'Value']] = df['Event'].str.extract(RE_PATTERN_EVENT, expand=True)
    df['Channel'] = df['Channel'].astype('category')
    df['Value'] = df['Value'].replace({'Low': 0, 'High': 1, 'Out': 0, 'In': 1})
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce', downcast='unsigned', dtype_backend='numpy_nullable')

    return df
