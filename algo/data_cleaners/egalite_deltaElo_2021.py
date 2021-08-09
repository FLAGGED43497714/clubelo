from algo.Elo_et_proba.Elo_and_HFA_update import one_match_update
import numpy as np
import pandas as pd
from algo.Elo_et_proba.Elo_and_HFA_update import one_match_update

path_matchs = "data/saison_2021_sans_cotes.csv"

Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]

nb_de_matchs = len(Matchs)
for match_nb in range(nb_de_matchs) :
    one_match_update(Matchs, currentElo, currentHFA, match_nb)
