import os.path as op
from glob import glob
import itertools
import numpy as np
from collections import defaultdict

# Parameters:
ds_factor = 8

# arrange the OLDT in the presentation order
subjects = {'A0023': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0085': ['OLDT2', 'SENT2', 'OLDT1'],
            # 'A0100': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0123': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0125': ['OLDT1', 'SENT2', 'OLDT2'],
            'A0127': ['OLDT2', 'SENT1', 'OLDT1'],
            'A0129': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0130': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0134': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0136': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0148': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0150': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0155': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0159': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0161': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0163': ['OLDT2', 'SENT2', 'OLDT1'],
            }

mne_bin = '/Applications/packages/mne-c/bin'
# drives
drives = {'nyu': op.join(op.expanduser('~'), 'Experiments', 'E-MEG', 'data'),
          'server': op.join('/Volumes', 'server', 'MORPHLAB', 'Teon',
                            'E-MEG', 'data'),
          'home': op.join('/Volumes', 'teon-backup', 'Experiments',
                          'E-MEG', 'data'),
          'office': op.join('/Volumes', 'GLYPH-1 TB', 'Experiments',
                            'E-MEG', 'data'),
          'google_drive': op.join(op.expanduser('~'), 'Google Drive',
                                  'E-MEG', 'data'),
          'dropbox': op.join(op.expanduser('~'), 'Dropbox', 'academic',
                             'Experiments', 'E-MEG', 'output'),
         }
results_dir = '/Users/teon/Dropbox/academic/Experiments/E-MEG/output/results/'

reject = dict(mag=3e-12)
img = 'png'

bads = defaultdict(lambda: ['MEG 130'])
bads['A0148'] = ['MEG 035', 'MEG 095', 'MEG 182', 'MEG 087']
bads['A0149'] = ['MEG 087', 'MEG 067', 'MEG 078', 'MEG 095',
                 'MEG 102', 'MEG 138', 'MEG 160', 'MEG 176', 'MEG 183',
                 'MEG 195']

# # mri dir
# 'mri_sdir': op.join('{mri_dir}', '{subject}'),
# 'label_sdir': op.join('{mri_sdir}', 'label'),

# # fwd model
# 'bem_head': op.join('{mri_sdir}', 'bem', '{subject}-head.fif'),
# 'bem': op.join('{mri_sdir}', 'bem', '{subject}-*-bem.fif'),
# 'bem-sol': op.join('{mri_sdir}', 'bem', '{subject}-*-bem-sol.fif'),
# 'src': op.join('{mri_sdir}', 'bem', '{subject}-ico-4-src.fif'),
# 'common_src': op.join('{mri_dir}', 'fsaverage', 'bem',
#                            'fsaverage-ico-4-src.fif'),


def kit2fiff(subject, exp, path, preload=False):
    from mne.io import read_raw_kit
    kit = op.join(path, subject, 'kit', 
                  subject + '*' + exp + '*' + 'calm.con')
    mrk_pre = op.join(path, subject, 'kit', 
                      subject + '*mrk*' + 'pre_' + exp + '*.mrk')
    mrk_post = op.join(path, subject, 'kit', 
                       subject + '*mrk*' + 'post_' + exp + '*.mrk')
    elp = op.join(path, subject, 'kit', subject + '*p.txt')
    hsp = op.join(path, subject, 'kit', subject + '*h.txt')

    mrk_pre = glob(mrk_pre)[0]
    mrk_post = glob(mrk_post)[0]
    elp = glob(elp)[0]
    hsp = glob(hsp)[0]
    kit = glob(kit)[0]

    # raw = read_raw_kit(input_fname=kit, mrk=[mrk_pre, mrk_post],
    #                    elp=elp, hsp=hsp, stim='>', slope='+',
    #                    preload=preload, verbose=False)

    raw = read_raw_kit(input_fname=kit, stim='>', slope='+',
                       preload=preload, verbose=False)

    return raw

def check_bad_chs(subject, raw, threshold=0.1, reject=4e-12, n_chan=5):
    """
    Check for flat-line channels or channels that repeatedly exceeded
    threshold.
    """
    basename = op.basename(raw.info['filename']).split('_')
    subject = basename[0]
    exp = basename[1]
    evt_file = op.join(op.dirname(raw.info['filename']), '..', 'mne', 
                       '_'.join([subject, exp + '-eve.txt']))
    evts = mne.load_events(evt_file)
    epochs = mne.Epochs(raw, tmin=-.2, tmax=.6, baseline=(None, 0),
                        reject={'mag': reject}, preload=True, verbose=False)
    threshold = epochs.events.shape[0] * threshold
    asdf
    if epochs.drop_log:
        bads = epochs.drop_log
        bads = E.table.frequencies(bads)
        bads = bads[bads['n'] > threshold]['cell'].as_labels()
    else:
        bads = []
    picks = mne.pick_types(epochs.info, exclude=[])
    data = epochs.get_data()[:, picks, :]
    flats = []
    diffs = np.diff(data) == 0
    for epoch in diffs:
        # channels flat > 50% time period
        flats.append(np.where(np.mean(epoch, 1) >= .5)[0])
    flats = np.unique(np.hstack(flats))
    flats = ['MEG %03d' % (x + 1) for x in flats]

    bad_chs = np.unique(np.hstack((bads, flats)).ravel())
    if len(bad_chs) > n_chan:
        drop = 1
    else:
        drop = 0

    with open(self.get('bads-file'), 'w') as FILE:
        import datetime
        date = datetime.datetime.now().ctime()
        FILE.write('# Log of bad channels for %s written on %s\n\n'
                   % (self.get('subject'), date))
        FILE.write('bads=%s\n' % bad_chs)
        FILE.write('drop=%s' % drop)
    return bad_chs