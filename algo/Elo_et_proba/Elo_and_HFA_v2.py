from Score_estimation1 import score_estimation



def elo_update_aftermatch(home_elo, away_elo, home_wins, K) :
    new_home_elo = home_elo + K * (home_wins - score_estimation(home_elo, away_elo) )
    new_away_elo = away_elo + K * ( (1 - home_wins) - score_estimation(away_elo, home_elo))

    return new_home_elo, new_away_elo



def one_match_update(Matchs, currentElo_home, currentElo_away, match_nb, K) :
    homeTeam = Matchs[match_nb][0]
    awayTeam = Matchs[match_nb][1]

    homeScore = int(Matchs[match_nb][2])
    awayScore = int(Matchs[match_nb][3])
        
    if homeScore > awayScore :
        home_wins = 1 
    if homeScore == awayScore :
        home_wins = 0.5 
    if homeScore < awayScore :
        home_wins = 0

    for k in range(len(currentElo_home)) :
        if currentElo_home[k][0] == homeTeam :
            homeElo = float(currentElo_home[k][1])
        if currentElo_away[k][0] == awayTeam :
            awayElo = float(currentElo_away[k][1])
        
    new_home_elo, new_away_elo = elo_update_aftermatch(homeElo, awayElo, home_wins, K)

    for k in range(len(currentElo_home)) :
        if currentElo_home[k][0] == homeTeam :
            currentElo_home[k][1] = str(new_home_elo)
        if currentElo_away[k][0] == awayTeam :
            currentElo_away[k][1] = str(new_away_elo)


