import tomllib as toml
from pathlib import Path
import numpy as np

from mne.stats import (permutation_cluster_1samp_test as pc_1samp_test)


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

task = cfg['task']
# parameters
random_state = 42
# decoding parameters
tmin, tmax = -.1, 1
n_folds = 5
# baseline
bmin, bmax = -.2, -.1
reject = cfg['reject']
c_names = ['word/target/primed', 'word/target/unprimed']
chance = .5

# setup group
fname_group_template = str(root / 'output' / 'group',
                           f'group_{task}_sensor_priming_%s.npy')

####################
# Group Statistics #
####################
group_scores = np.load(fname_group_template % 'scores')

########################
# run a TD clustering  #
########################
# remove chance from the time decoding
group_scores -= chance
group_stats = pc_1samp_test(group_scores, n_permutations=10000,
                            threshold=1.96, tail=0,
                            seed=42, n_jobs=-1)

# unpacking significant time bounds for Time Decoding
T_obs, clusters, p_values, _ = group_stats

good_cluster_inds = np.where(p_values < 0.05)[0]

times = np.load(fname_group_template % 'times')
td_sig = np.zeros(len(times))
idx_time_sig = list()
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster infomation, get unique indices
    td_sig[clusters[clu_idx]] = 1
    if not isinstance(clu_idx, list):
        clu_idx = [clu_idx]
    for idx in clu_idx:
        np.indices(times.shape)
        interval = clusters[idx][0]
        if interval[1] == len(times):
            idx = [interval[0], interval[1] - 1]
        else:
            idx = [interval[0], interval[1]]
        idx_time_sig.append(idx)

# #######################
# # Group Time Decoding #
# #######################

# # individual td plots
# fig, axes = plt.subplots(5, 4, figsize=(20, 10))
# axes[-1, -1].axis('off')
# axes = [ax for axis in axes for ax in axis]
# for subject, scores, ax in zip(subjects, group_scores, axes):
#     pretty_decod(scores=scores, chance=chance, sfreq=sfreq, ax=ax, times=times)
#     ax.set_title(subject)
# fig.tight_layout()
# group_rep.add_figs_to_section(fig, 'Individual TDs', 'Individual Plots',
#                                 image_format=img)

# # individual td patterns
