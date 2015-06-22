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
    fname_cov = op.join(config.drive, subject, 'mne',
                        subject +'_%s_calm_%s_filt-cov.fif' % (exp, filt))
    fname_fwd = op.join(config.drive, subject, 'mne',
                        subject + '_%s-fwd.fif' % exp)
    fname_inv = op.join(config.drive, subject, 'mne',
                        subject + '_%s-inv.fif' % exp)

    if not op.exists(fname_inv) or redo:
        # COV
        cov = mne.read_cov(fname_cov)
        # INV OP
        fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)
        info = mne.io.read_info(fname_epo)
        inv_op = mne.minimum_norm.make_inverse_operator(info, fwd, cov)
        mne.minimum_norm.write_inverse_operator(fname_inv, inv_op)
