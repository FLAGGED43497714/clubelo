import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd

path_matchs = "data\saisons\saison_2021_sans_cotes.csv"
path_start_elo = "data\elo_start\elo_start_2020_v3.csv"

match_prediction_draw = draw_proba_writer(path_matchs, path_start_elo)

nb_of_matchs = len(match_prediction_draw)

for match_nb in range(nb_of_matchs) :
    delta_elo = float(match_prediction_draw[match_nb][2])
    perct_restant = 1 - float(match_prediction_draw[match_nb][4])
    Eh = score_estimation2(delta_elo)
    home_percent = ( 1 - Eh ) * perct_restant
    away_percent = Eh * perct_restant
    match_prediction_draw[match_nb][3] = str(home_percent)
    match_prediction_draw[match_nb][5] = str(away_percent)

df = pd.DataFrame(match_prediction_draw)
df.to_csv("fullversion1.csv", index=False, header = False)
