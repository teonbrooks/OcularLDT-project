import inspect
import os
import os.path as op
import fa


# with open('/Users/teon/Desktop/cleaned.txt') as FILE:
#     words = FILE.readlines()
#     words = [x.strip() for x in words]

free = fa.Free()
words = set([x['CUE'] for x in free._free if x['QPS'] == 'N'])

stims = []
missing = []
for word in words:
    entries = free[word]
    if entries:
        entries = [x for x in entries if x['TPS'] == 'V']
        if entries:
            targets = [x['TARGET'].lower() for x in entries]
            if len(targets) > 1:
                strengths = [entry['FSG'] for entry in entries]
                ranks = [(i, rank) for i, rank in enumerate(strengths)]

                high = entries[strengths.index(max(strengths))]['TARGET']
                high_s = max(strengths)
                low = entries[strengths.index(min(strengths))]['TARGET']
                low_s = min(strengths)
                best = [high, high_s, low, low_s]
                idx = [x[0] for x in sorted(ranks, key=lambda x: x[1])]
                from operator import itemgetter
                stim = [word.upper()]
                stim.extend(best)
                targets = itemgetter(*idx)(targets)
                ranks = itemgetter(*idx)(strengths)
                complete = []
                for vals in zip(targets, ranks):
                    stim.extend(vals)
                stims.append(stim)
            else:
                missing.append(word)
        else:
            missing.append(word)
    else:
        missing.append(word)

with open('/Users/teon/Desktop/sentences-verbs-complete.txt', 'wb') as FILE:
    for stim in stims:
        FILE.write('\t'.join(stim) + '\n')
