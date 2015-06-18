import mne
import os.path as op
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'


for subject in config.subjects:
    print subject

    path = op.join(config.drives[drive], subject, 'mne')
    ep_fname = op.join(path, '%s_%s_ica_calm_filt-epo.fif'
                       % (subject, exp))
    cov_fname = op.join(path, op.basename(ep_fname)[:10] + '-cov.fif')

    if not op.exists(cov_fname):
        r = Report()
        report_fname = op.join(op.expanduser('~'), 'Dropbox', 'academic', 
                               'Experiments', 'E-MEG', 'output', 'results',
                               subject, '%s_%s_filt_cov-report.html' % (subject, exp))

        epochs = mne.read_epochs(ep_fname)
        epochs.pick_types(meg=True, exclude='bads')

        # Read projection
        proj_file = op.join(path, op.basename(ep_fname)[:10] + '-proj.fif')
        projs = [mne.read_proj(proj_file)[0]]

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Evoked Response'}, show=False)
        r.add_figs_to_section(p, 'Evoked Response to Prime Word',
                              'Evoked', image_format=img)

        # plot covariance and whitened evoked
        epochs.crop(-.2, -.1, copy=False)
        cov = mne.compute_covariance(epochs, projs=projs, method='auto',
                                     verbose=False)
        p = cov.plot(epochs.info, show_svd=0, show=False)[0]
        comments = ('The covariance matrix is computed on the -200:-100 ms '
                    'baseline. -100:0 ms is confounded with the eye-mvt.')
        r.add_figs_to_section(p, 'Covariance Matrix', 'Covariance',
                              image_format=img, comments=comments)
        p = evoked.plot_white(cov, show=False)
        r.add_figs_to_section(p, 'Whitened Evoked to Prime Word', 'Covariance',
                              image_format=img)

        r.save(report_fname, overwrite=True, open_browser=False)

        # save covariance
        mne.write_cov(cov_fname, cov)
