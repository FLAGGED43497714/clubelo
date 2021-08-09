import numpy as np
import pandas as pd

path = "data/2020-08-20"
matchs_v1 = np.genfromtxt(path, delimiter=',', dtype=str)

matchs_v2 = [[]]
for k in range(len(matchs_v1)) :
    if (matchs_v1[k][2] == 'FRA') and (matchs_v1[k][3] == '1') :
            matchs_v2.append(matchs_v1[k])


df = pd.DataFrame(matchs_v2)
df.to_csv('data/elo_start_2020.csv')
