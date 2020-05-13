import pickle
import os.path as op

import config
from plot_func import group_plot


# parameters
subjects = config.subjects
path = config.drive
results_dir = config.results_dir
exp = config.exp
filt = config.filt
clf_name = 'ridge'
analysis = 'freq_%s_regression_sensor_analysis' % clf_name
c_names = 'freq'
title = 'Word Frequency'
threshold = 1.96
p_accept = 0.05
chance = .5
reg_type = 'reg'

# setup group
group_template = op.join('%s', 'group', 'group_%s_%s_filt_%s.%s')
fname_group_rep = group_template % (results_dir, exp, filt, analysis, 'html')

group_rep = group_plot(subjects, path, results_dir, exp, filt, clf_name,
                       analysis, c_names, title, threshold, p_accept, chance,
                       img='png', reg_type=reg_type)

group_rep.save(fname_group_rep, open_browser=True, overwrite=True)
