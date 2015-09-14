evts
unique_stimes, idx = np.unique(evts[:, 0], return_inverse=True)
mappings = list()
for ii, stime in enumerate(unique_stimes):
    triggers = np.sort(evts[np.where(ii == idx)[0], -1])
    mappings.append([unique_stimes[ii], triggers])

event_id
# 1: nonword, 2: word, 3: unprimed, 4: primed, 5: prime, 6: target,
# 50: alignment, 99: fixation
