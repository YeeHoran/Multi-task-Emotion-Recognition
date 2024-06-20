import pandas as pd
import numpy as np
import math

df = pd.read_csv('./data/publictest-20240609-2-yh.csv')
columnV = df['Valence']
i = 0
for v in columnV:
    i += 1
    if np.isnan(v):
        print(i)
