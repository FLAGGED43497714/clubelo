import numpy as np
import pandas as pd

path_matchs = "data\saisons\saison_2021_sans_cotes.csv"
path_out = "testfile69.csv"


Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]

nb_de_matchs = len(Matchs)

def score(a,b) :
    if a > b :
        return 1 
    if a == b :
        return 2
    if a < b :
        return 3

res = [[Matchs[k][0], Matchs[k][1], score(Matchs[k][2],Matchs[k][3])]  for k in range(nb_de_matchs) ]

df = pd.DataFrame(res)
df.to_csv(path_out, index=False, header = False)
