import os.path as op


directory = '/Applications/packages/E-MEG/input/stim/Exp2/'

with open(op.join(directory, 'Exp2_stimuli', 'Experimental_List-Table 1.csv'),
          'rb') as FILE:
    header = FILE.readline()
    header = header.split(',')
    header_line = header[:2]
    header_line.append('Target')
    header_line.append(header[4] + '\n')
    header = '\t'.join(header_line)
    doc = FILE.readlines()[1:]
    doc = [line.split(',')[:5] for line in doc]
primes = [words[0][0].upper() + words[0][1:].lower() for words in doc]
verbs = [words[1].lower() for words in doc]
targets_high = [words[2].lower() for words in doc]
targets_low = [words[3].lower() for words in doc]
fillers = [words[4].lower() + ".\n" for words in doc]

list1 = []
list2 = []
for a, b in zip(*(targets_high[::2], targets_low[1::2])):
    list1.extend([a,b])
for a, b in zip(*(targets_low[::2], targets_high[1::2])):
    list2.extend([a,b])
sentences_l1 = zip(*(primes, verbs, list1, fillers))
sentences_l2 = zip(*(primes, verbs, list2, fillers))

with open(op.join(directory, 'List1.txt'), 'w') as TARGET1:
    TARGET1.write(header)
    for sentence in sentences_l1:
        sentence = '\t'.join(sentence)
        TARGET1.write(sentence)

with open(op.join(directory, 'List2.txt'), 'w') as TARGET2:
    TARGET2.write(header)
    for sentence in sentences_l2:
        sentence = '\t'.join(sentence)
        TARGET2.write(sentence)