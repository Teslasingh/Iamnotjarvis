import os,sys,platform
from pathlib import Path
import re
home = Path.home()
paths = [home]
# Common folders
candidates = [home/'Documents', home/'Desktop', home/'Downloads']
paths = [p for p in candidates if p.exists()]
pattern = re.compile(r'(?i)(interview)[\s._-]*(assignment)|(assignment)[\s._-]*(interview)')
results = []
for base in paths:
    for root, dirs, files in os.walk(base, followlinks=False):
        for name in files:
            try:
                if pattern.search(name):
                    p = Path(root)/name
                    st = p.stat()
                    results.append((str(p), st.st_size, st.st_mtime))
            except Exception:
                pass
if not results:
    print('NO MATCHES FOUND')
else:
    for p,size,mt in sorted(results):
        print(f"{p} | {size} bytes | {__import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime(mt))}")
