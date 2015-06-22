import subprocess
import os


files = ['001-make_events.py',
         '002-make_epochs.py',
         '003a-make_pca.py',
         '004-make_cov.py',
         # '005-make_fwd.py',
         # '006-make_inv.py',
         '101a-get_target_times.py',
         '102-make_design_matrix.py']

for FILE in files:
    cmd = ['python', FILE]        
    cwd = '/Applications/packages/E-MEG/scripts/'
    # sp = subprocess.call(cmd, cwd = cwd)
    #
    sp = subprocess.Popen(cmd, cwd=cwd,
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()

    if stderr:
        print '\n> ERROR:'
        print '%s\n%s' %(stderr, stdout)
