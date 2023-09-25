from copy import copy
import numpy as np
from pandas import DataFrame, concat

from pyeparse import read_raw, RawEDF


"""Terminology
The definitions for the Reading class methods are derived from [citation].
"""
terms = {
         'index': 'The fixation order.',
         'eye': '`0` corresponds to the left eye, `1` corresponds to the right,'
                ' and `2` corresponds to binocular tracking.',
         'stime': 'Start time of fixation.',
         'etime': 'End time of fixation.',
         'axp': 'Average x-position',
         'ayp': 'Average y-position',
         'dur': 'Duration of fixation: `etime` - `stime`.',
         'fix_pos': 'Index of the interest area.',
         'trial': 'Trial Number.',
         'max_pos': 'Maximum position in a sequence.',
         'is_gaze': 'Boolean of the first time(s) fixating on a given item in a '
                 'sequence. Must also be the max position in the sequence.',
         'is_first_fix': 'Boolean of the very first fixation on a given item in a '
                   'sequence. Must also be the max position in the sequence.'
        }

class AOIReport(object):
    """ Create interest area summaries for Raw

    Parameters
    ----------
    raw : filename | instance of Raw
        The filename of the raw file or a Raw instance to create AOIReport
        from.
    ias : str | ndarray (n_ias, 7)
        The interest areas. If str, this is the path to the interest area file.
        Only .ias formats are currently supported. If array, number of row must
        be number of interest areas. The columns are as follows:
        # RECTANGLE id left top right bottom [label]
    ia_labels : None | list of str
        Interest area name. If None, the labels from the interest area file
        will be used.
    depmeas : str
        Dependent measure. 'fix' for fixations, 'sac' for saccades.
    trial_msg : str
        The identifying tag for trial. Default is 'TRIALID'.

    Returns
    -------
    AOIReport : an instance of AOIReport
        An Interest Areas report. Fixations or Saccades are limited to the areas
        described in `ias`. The fix/sacc are labeled in a given region
        per trial.
        Info is stored for Trial x IAS x fixations/saccades
    """
    def __init__(self, raw, ias, ia_labels=None):

        if isinstance(raw, str):
            raw = read_raw(raw)
        elif not isinstance(raw, RawEDF):
            raise TypeError('raw must be Raw instance of filename, not %s'
                            % type(raw))

        if isinstance(ias, str):
            ias = read_ia(ias)
        self.n_ias = ias.shape[0]
        if ia_labels is not None:
            ias[:, -1] = ia_labels
        self._ias = dict([(ii[-1], int(ii[1])) for ii in ias])
        self.ia_labels = sorted(self._ias, key=self._ias.get)

        self.terms = terms

        df = DataFrame(raw.discrete['messages'])
        df = df.astype({'msg': 'str'}, copy=False)
        
        # Add TTL timestamp
        df_ttl = df[df['msg'].str.startswith('TTL')].copy()
        df_ttl['metric'] = 'ttl'
        # Add Trial Start timestamp
        df_trial = df[df['msg'].str.contains('TRIALID')].copy()
        df_trial['metric'] = 'trial_start'
        self.n_trials = df_trial.shape[0]
        # Add Invisible Boundary Message
        pat = 'INVISIBLE_BOUNDARY_[A-Za-z]*_IN'
        df_boundary = df[df['msg'].str.contains(pat)].copy()
        df_boundary['metric'] = 'boundary'
        # Add an end of recording trial to make computation easier
        last = df_trial.iloc[-1].copy()
        last['stime'] += 20.
        last['msg'] = 'End of Recording'
        df_trial = concat((df_trial, last.to_frame().T)).reset_index(drop=True)
        # Add fixation data
        df_fix = DataFrame(raw.discrete['fixations'])
        df_fix['metric'] = 'fixation'
        # Add saccade data
        df_sac = DataFrame(raw.discrete['saccades'])
        df_sac['metric'] = 'saccade'
        # Add blink data
        df_blink = DataFrame(raw.discrete['blinks'])
        df_blink['metric'] = 'blink'
        # Add button data
        df_button = DataFrame(raw.discrete['buttons'])
        df_button['metric'] = 'btn_press'
        
        data = concat([df_sac, df_fix, df_trial, df_ttl,
                       df_boundary, df_blink, df_button],
                       axis=0).reset_index(drop=True)
        data['msg'] = data['msg'].fillna('').astype(str)
        data.sort_values('stime', inplace=True)

        trial_times = np.array((df_trial['stime'].to_numpy(),
                               df_trial['stime'].shift(-1).to_numpy()), 
                               dtype='float').T
        trial_durations = df_trial['stime'].shift(-1) - df_trial['stime']
        self.trial_durations = trial_durations.tolist()[:-1]

        # adding new columns, initializing to -1
        # fix_pos is the IA number
        boundary_pos = np.negative(np.ones(len(data), int))
        fix_pos = np.negative(np.ones(len(data), int))
        trial_no = np.negative(np.ones(len(data), int))

        for ii, meas in enumerate(data.iterrows()):
            _, meas = meas
            stime = meas['stime']
            idx = ((trial_times[:,0][:, np.newaxis] <= stime) &
                   (stime <= trial_times[:,1][:, np.newaxis]))
            trial_no[ii] = np.where(idx)[0][0]

            if 'INVISIBLE' in meas['msg']:
                if 'FIX' in meas['msg']:
                    boundary_pos[ii] = fix_pos[ii] = 0
                elif 'PRIME' in meas['msg']:
                    boundary_pos[ii] = fix_pos[ii] = 1
                elif 'TARGET' in meas['msg']:
                    boundary_pos[ii] = fix_pos[ii] = 2
                elif 'POST' in meas['msg']:
                    boundary_pos[ii] = fix_pos[ii] = 3

            if meas['metric'] not in ('fixation', 'saccade'):
                continue

            prefix = 'a' if meas['metric'] == 'fixation' else 's'
            xp, yp = prefix + 'xp', prefix + 'yp'

            # RECTANGLE id left top right bottom [label]
            for jj, ia in enumerate(ias):
                _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                if int(ia_left) < int(meas[xp]) < int(ia_right) \
                and int(ia_top) < int(meas[yp]) < int(ia_bottom):
                    fix_pos[ii] = jj
                    break

        dur = data['etime'] - data['stime']
        aoi = DataFrame({'trial_no': trial_no,
                         'dur': dur,
                         'fix_pos': fix_pos,
                         'boundary_pos': boundary_pos})
        data = concat([data, aoi], axis=1).reset_index(drop=True)

        # # drop all fixations outside of the interest areas
        # data = data[data['fix_pos'].astype(int) >= 0]
        # data = data.reset_index()
        # data.fix_pos = data.fix_pos.astype(int)

        df_gaze = self._define_gaze(data)
        data = concat((data, df_gaze), axis=1).reset_index(drop=True)

        # DataFrame
        self.data = data

    def __repr__(self):
        ee, ii = self.n_trials, self.n_ias

        return '<AOIReport | {0} Trials x {1} IAs>'.format(ee, ii)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return (self.n_trials, self.n_ias)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx not in self._ias:
                raise KeyError("'%s' is not found." % idx)
            data = self.data[self.data['fix_pos'] == self._ias[idx]]
            return data
        elif isinstance(idx, int):
            idx = self.data['trial_no'] == idx
            data = self.data[idx]
            return data
        elif isinstance(idx, slice):
            inst = copy(self)
            inst.data = self.data[idx]
            inst.n_trials = len(inst.data)
            return inst
        else:
            raise TypeError('index must be an int, string, or slice')
    
    def _define_gaze(self, data):

        # # create a max position using lag
        # max_pos is the maximum IA visited
        max_pos = np.negative(np.ones(len(data), int))
        max_boundary_pos = np.negative(np.ones(len(data), int))
        max_pos[0] = data.iloc[0]['fix_pos']
        for idx, meas in enumerate(data.iloc[1:].iterrows(), 1):
            _, meas = meas
            # propagate boundary info
            if meas['metric'] == 'boundary':
                max_boundary_pos[idx] = meas['boundary_pos']
            elif meas['trial_no'] == data.loc[idx - 1, 'trial_no']:
                max_boundary_pos[idx] = max_boundary_pos[idx - 1]
            
            if meas['trial_no'] != data.loc[idx - 1, 'trial_no']:
                max_pos[idx] = meas['fix_pos']
            else:
                if meas['fix_pos'] > max_pos[idx - 1]:
                    max_pos[idx] = meas['fix_pos']
                else:
                    max_pos[idx] = max_pos[idx - 1]

        # initializing
        gaze = np.zeros(len(data), int)
        gaze_ias = np.zeros(self.n_ias)
        ref_trial_no = data.iloc[0]['trial_no']
        first_fix = np.zeros(len(data), int)

        for idx, meas in enumerate(data.iloc[1:].iterrows(), 1):
            _, meas = meas
            fix_pos = meas['fix_pos']
            prev_fix_pos = data.iloc[idx - 1]['fix_pos']

            if meas['trial_no'] > ref_trial_no:
                gaze_ias = np.zeros(self.n_ias, int)
                ref_trial_no = meas['trial_no']
            if all((fix_pos > 0, fix_pos >= max_pos[idx])):
                if gaze_ias[fix_pos] == 0:
                    gaze_ias[fix_pos] = 1
                    gaze[idx] = 1
                    first_fix[idx] = 1
                elif gaze[idx - 1] == 1 and fix_pos == prev_fix_pos:
                    gaze[idx] = 1

        df_gaze = DataFrame({'max_pos': max_pos,
                             'max_boundary_pos': max_boundary_pos,
                             'gaze': gaze,
                             'first_fix': first_fix})
        return df_gaze

def read_ia(filename):
    with open(filename) as FILE:
        ia = FILE.readlines()
    idx = [i for i, line in enumerate(ia) if line.startswith('Type')]
    if len(idx) > 1:
        raise IOError('Too many headers provided in this file.')
    ias = ia[idx[0]+1:]
    ias = np.array([line.split() for line in ias])

    return ias
