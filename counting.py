import numpy as np
import pandas as pd
from pathlib import Path


for pp in sorted(Path('.').rglob('labelled_summary.csv')):
    print(pp)
    if str(pp).find('test') != -1 or str(pp).find('scratch') != -1:
        continue

    df = pd.read_csv(str(pp))
    for i in range(20):
        ct = 0
        for vv in  df.iloc[i,:].to_dict().values():
            if type(vv) == str:
                continue
            if np.isclose(vv,0):
                ct += 1
        print(ct,end=' ')

    print()
