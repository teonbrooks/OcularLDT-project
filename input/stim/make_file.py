from glob import glob
import inspect

file = os.path.abspath(inspect.getfile(inspect.currentframe()))
path = os.path.dirname(file)
os.chdir(path, 'src')

compiled = os.path.join(path, 'free_association.txt')

proto = []
for f in glob('*.html'):
    f = open(f).readlines()
    drop = []
    f = [line for i, line in enumerate(f) if not line.startswith('<')]
    proto.extend(f)
proto = ''.join(proto)

with open(compiled, 'w') as g:
    g.write(proto)