import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd

# path_matchs = "data\saisons\\2020_2021_avec_v4.csv"
# path_start_elo = "data\elo_start\elo_start_2020_v3.csv"

path_matchs = "data\saisons\\saison_2019_2020_sans_cotes.csv"
path_start_elo = "data\elo_start\elo_start_2019_v2.csv"


path_out = "fullversion1.csv"

# match_prediction_draw = draw_proba_writer(path_matchs, path_start_elo, imported=True)
match_prediction_draw = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
path_import_W1="data/NN_saved/W1_BR100n1", path_import_W2="data/NN_saved/W2_BR100n1")

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
df.to_csv(path_out, index=False, header = False)
