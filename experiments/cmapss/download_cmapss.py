from pathlib import Path
from urllib.request import urlopen
import hashlib

OUT = Path(__file__).resolve().parent / 'cmapss'
BASE = 'https://raw.githubusercontent.com/huster123/c-mapss-full-dataset-/master/Data/'
FILES = ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt']

for name in FILES:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE + name, timeout=60) as src, target.open('wb') as dst:
        dst.write(src.read())
    print(name, target.stat().st_size, hashlib.sha256(target.read_bytes()).hexdigest())

