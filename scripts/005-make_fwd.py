import os.path as op
import mne
import config


path = op.join(config.drive, '..', 'MRI')
exp = 'OLDT'
filt = config.filt
redo = False

for subject in config.subjects:
    print config.banner % subject

    # Define filenames
    fname_epo = op.join(config.drive, subject, 'mne', 
                        subject + '_%s_xca_calm_%s_filt-epo.fif'
                        % (exp, filt))
    fname_trans = op.join(config.drive, subject, 'mne',
                          subject +'-trans.fif')
    fname_fwd = op.join(config.drive, subject, 'mne',
                        subject + '_%s-fwd.fif' % exp)
    bem_sol = op.join(path, subject, 'bem',
                      subject + '-inner_skull-bem-sol.fif')
    src = op.join(path, subject, 'bem', subject + '-ico-4-src.fif')

    if not op.exists(fname_fwd) or redo:
        info = mne.io.read_info(fname_epo)

        fwd = mne.make_forward_solution(info=info, trans=fname_trans, src=src,
                                        bem=bem_sol, fname=fname_fwd,
                                        meg=True, eeg=False,
                                        mindist=0.0, ignore_ref=True)