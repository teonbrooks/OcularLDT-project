import os.path as op

import mne
from mne.report import Report

from mne_bids import read_raw_bids
from mne_bids.utils import get_entity_vals


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'Experiments', task)

subjects_list = get_entity_vals(bids_root, entity_key='sub')

for subject in subjects_list[:1]:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)
    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    fname_epo = op.join(path, subject + '_%s_xca_calm_%s_filt-epo.fif'
                        % (exp, filt))
    fname_cov = op.join(path, subject + '_%s_calm_%s_filt-cov.fif'
                        % (exp, filt))
    fname_proj = op.join(path, subject + '_%s_calm_%s_filt-proj.fif'
                         % (exp, filt))
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s_filt_cov-report.html' % (exp, filt))

    if not op.exists(fname_cov) or redo:
        rep = Report()
        epochs = mne.read_epochs(fname_epo)
        epochs.info['bads'] = config.bads[subject]
        epochs.pick_types(meg=True, exclude='bads')

        # back to coding
        proj = mne.read_proj(fname_proj)
        epochs.add_proj(proj)
        epochs.apply_proj()

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Evoked Response'}, show=False)
        rep.add_figs_to_section(p, 'Evoked Response to Prime Word',
                              'Evoked', image_format=img)

        # plot covariance and whitened evoked
        epochs.crop(-.2, -.1, copy=False)
        cov = mne.compute_covariance(epochs, method='auto', verbose=False)
        p = cov.plot(epochs.info, show_svd=0, show=False)[0]
        comments = ('The covariance matrix is computed on the -200:-100 ms '
                    'baseline. -100:0 ms is confounded with the eye-mvt.')
        rep.add_figs_to_section(p, 'Covariance Matrix', 'Covariance',
                              image_format=img, comments=comments)
        p = evoked.plot_white(cov, show=False)
        rep.add_figs_to_section(p, 'Whitened Evoked to Prime Word', 'Covariance',
                              image_format=img)

        rep.save(fname_rep, overwrite=True, open_browser=False)

        # save covariance
        mne.write_cov(fname_cov, cov)
