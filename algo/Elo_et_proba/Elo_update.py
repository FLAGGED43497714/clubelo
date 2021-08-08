from Score_estimation1 import score_estimation

with open("clubelo\config\k_value.txt") as f :
    K =  int(f.read())

def elo_update(home_elo, away_elo, home_wins) :
    new_home_elo = home_elo + K * (home_wins + score_estimation(team=home_elo, opponent=away_elo) )

    new_away_elo = away_elo + K * ( (1 - home_wins) + score_estimation(team_elo=away_elo, opponent_elo=home_elo))

    return new_home_elo, new_away_elo



# faire les ajustement en fonction du score 