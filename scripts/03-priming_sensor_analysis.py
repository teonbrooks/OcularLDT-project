import os.path as op
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.report import Report
import config

prep = 'calm_lp40'


for subject in config.subjects:
    path = op.join(config.data_dir, subject, 'mne')
    r = Report()
    r_path = op.join(config.results_dir, subject, 'meg',
                     '%s_OLDT_priming_sensor_analysis.html' % subject)
    for exp in config.subjects[subject]:
        if not exp.startswith('OLDT'):
            continue
        else:
            # load raw
            raw_file = op.join(path, '%s_%s_%s-raw.fif' % (subject, exp, prep))
            raw = mne.io.read_raw_fif(raw_file, verbose=False, preload=False)
            raw.info['bads'] = config.bads[subject]
            try:
                proj = mne.read_proj(raw_file[:-18] + '-proj.fif')
                raw.add_proj(proj).apply_proj()
            except IOError:
                warnings.warn('No projector applied to subject %s.' % subject)
            evts = mne.read_events(op.join(path, '%s_%s-eve.txt' % (subject, exp)))
            event_id = {'unprimed': 3, 'primed': 4}
            # load epochs
            epochs = mne.Epochs(raw, evts, event_id, tmin=-.1, tmax=.5,
                                baseline=(None,0), reject=config.reject,
                                preload=True, decim=2, verbose=False)
            # plotting grand average
            p = epochs.average().plot(show=False)
            r.add_figs_to_section(p, '%s: Grand Average on Target' % exp,
                                  'Summary', image_format='png')
            # compute/plot difference
            epochs_list = [epochs[k] for k in ('unprimed', 'primed')]
            mne.epochs.equalize_epoch_counts(epochs_list)
            evoked = epochs_list[1].average() - epochs_list[0].average()
            p = evoked.plot(show=False)
            r.add_figs_to_section(p, '%s: Difference Butterfly' % exp,
                                  'Evoked Difference Comparison',
                                  image_format='png')
            p = evoked.plot_topomap(np.linspace(0, .25, 10), show=False)
            r.add_figs_to_section(p, '%s: Difference Topomap 0-250 ms' % exp,
                                  'Evoked Difference Comparison',
                                  image_format='png')
            p = evoked.plot_topomap(np.linspace(.25, .50, 10), show=False); 
            r.add_figs_to_section(p, '%s: Difference Topomap 250-500 ms' % exp,
                                  'Evoked Difference Comparison',
                                  image_format='png')
            # get ready for decoding ;)
            n_times = len(epochs.times)
            data_picks = mne.pick_types(epochs.info, meg=True, exclude='bads')
            X = [e.get_data()[:, data_picks, :] for e in epochs_list]
            y = [k * np.ones(len(this_X)) for k, this_X in enumerate(X)]
            X = np.concatenate(X).astype('float32')
            y = np.concatenate(y).astype('float32')

            from sklearn.svm import SVC
            from sklearn.cross_validation import cross_val_score, ShuffleSplit

            clf = SVC(C=1, kernel='linear')
            # Define a monte-carlo cross-validation generator (reduce variance):
            cv = ShuffleSplit(len(X), 10, test_size=0.2)

            scores = np.empty(n_times, np.float32)
            std_scores = np.empty(n_times, np.float32)

            for t in xrange(n_times):
                Xt = X[:, :, t]
                # Standardize features
                Xt -= Xt.mean(axis=0)
                Xt /= Xt.std(axis=0)
                # Run cross-validation
                # Note : for sklearn the Xt matrix should be 2d (n_samples x n_features)
                scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=1)
                scores[t] = scores_t.mean()
                std_scores[t] = scores_t.std()

            times = 1e3 * epochs.times
            scores *= 100  # make it percentage
            std_scores *= 100

            plt.close('all')
            fig = plt.figure()
            plt.plot(times, scores, label="Classif. score")
            plt.axhline(50, color='k', linestyle='--', label="Chance level")
            plt.axvline(0, color='r', label='stim onset')
            plt.legend()
            hyp_limits = (scores - std_scores, scores + std_scores)
            plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
                             color='b', alpha=0.5)
            plt.xlabel('Times (ms)')
            plt.ylabel('CV classification score (% correct)')
            plt.ylim([30, 100])
            plt.title('Sensor space decoding')

            r.add_figs_to_section(fig, '%s: Decoding Score on Priming' % exp,
                                  'Decoding', image_format='png')
        r.save(r_path, overwrite=True)
