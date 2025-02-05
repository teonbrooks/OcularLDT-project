"""
TODO: this script is not DRY (cf. ../_to_bids/_recode_events.py)
Consider a consolidation, perhaps a helper file.

Recoding Events
---------------
To deal with the way MNE deals with events with multiple tagging, we
devised a way to handle this by repacking tag info in a binary fashion.
Maybe next time I will code my events just directly as such
(consult schematic in lab notebook).

Prime += 1
Target += 2
Priming += 4
Nonword += 8

This recoding efficiency allows for Hierarchial Event Descriptors
(HED tags).
"""
import numpy as np


def recode_events(triggers):
    
    expt_idx = np.array([(x & 2 ** 5) >> 5 for x in triggers], dtype=bool)

    # unpack trigger information
    semantic = np.array([(x & 2 ** 4) >> 4 for x in triggers],
                        dtype=bool)
    nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2
                            for x in triggers])
    current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0
                            for x in triggers])

    idx = np.where(nonword_pos - current_pos != 0)[0]
    # word region
    idy = np.where(np.logical_and(current_pos > 0,
                                  current_pos < nonword_pos))[0]
    # nonword is absent
    idy2 = np.where(nonword_pos == 0)[0]
    # not fixation cross
    idy3 = np.where(current_pos != 0)[0]
    idy = np.unique(np.hstack((idy, idy2, idy3)))
    # find nonwords
    n_idx = np.where(nonword_pos - current_pos == 0)[0]
    n_idy = np.where(nonword_pos != 0)[0]

    # nonwords vs words
    words_idx = np.intersect1d(idx, idy)
    nonwords_idx = np.intersect1d(n_idx, n_idy)
    # fixation
    fix_idx = np.where(current_pos == 0)[0]
    # prime vs target
    primes_idx = np.where(current_pos == 1)[0]
    targets_idx = np.where(current_pos == 2)[0]
    # semantic
    semantic_idx = np.where(semantic)[0]

    ############
    # Recoding #
    ############
    recodeds = np.zeros(triggers.shape[0], int)
    # prime vs target
    recodeds[primes_idx] += 1
    recodeds[targets_idx] += 2
    # semantic priming
    recodeds[semantic_idx] += 4
    # word vs nonword
    recodeds[nonwords_idx] += 8
    # fixation
    recodeds[fix_idx] = 128
    # make all practice trials zero
    recodeds[~expt_idx] = 0

    # new_trig = {'fix_pos': current_pos,
    #             'old_trig': triggers,
    #             'trig': recodeds}
    
    return recodeds

def find_eyetrack_events(raw, pattern):
    """Find messages already parsed

    Parameters
    ----------
    raw : instance of eyelinkio.RawEDF
        the raw file to find events in.
    pattern : str | callable
        A substring to be matched or a callable that matches
        a string, for example ``lambda x: 'my-message' in x``
    event_id : int
        The event id to use.

    Returns
    -------
    idx : instance of numpy.ndarray (times, event_id)
        The indices found.
    """
    df = raw.discrete['messages']
    if callable(pattern):
        func = pattern
    elif isinstance(pattern, str):
        def func(x):
            return pattern in x
    else:
        raise ValueError('Pattern not valid. Pass string or function')
    # may not need to decode string in eyelinkio>=0.4.0
    idx = np.array([func(msg.decode('ASCII')) for msg in df['msg']])
    msg = df['msg'][idx]

    return msg