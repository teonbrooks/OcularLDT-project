# do a rms per epoch and overlay time for demonstrative purposes
# fig = mne.viz.plot_epochs_image(epochs, order=np.argsort(durs),
#                                 overlay_times=durs)
figs.append(fig)
# plotting grand average

p = epochs.average().plot(show=False)
comment = ("This is a grand average over all the target epochs.<br>"
           'Number of epochs: %d.' % (len(epochs)))
rep.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                        'Summary', image_format=img, comments=comment)


asdf
# s = stats[names[-1]].mlog10_p_val
s = stats[names[-1]].t_val
# plot p-values
interval = int(plt_interval * 1e3 / decim)   # plot every 5ms
times = epochs.times[::interval]
figs = list()
for time in times:
    # figs.append(s.plot_topomap(time, vmin=0, vmax=3, unit='',
    #                            scale=1, cmap='Reds', show=False))
    figs.append(s.plot_topomap(time, vmin=1, vmax=4, unit='',
                               scale=1, cmap='Reds', show=False))
    plt.close()
# rep.add_slider_to_section(figs, times, 'Regression Analysis (-log10 p-val)')
rep.add_slider_to_section(figs, times, 'Regression Analysis (t-val)')
rep.save(fname_rep, open_browser=False, overwrite=True)
