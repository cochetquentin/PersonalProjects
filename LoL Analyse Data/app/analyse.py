from tools import *

# Load the data
url = "https://gol.gg/tournament/tournament-matchlist/LEC%20Winter%20Season%202024/"
match_infos, game_stats = get_all_match_data_from_gol_url(url)

#Plot the data
figs = []
figs.append(plot_side_wr(match_infos))
figs.append(plot_side_wr_by_team(match_infos))
figs.append(plot_average_stats_by_team(match_infos))
figs.append(plot_drakes_taken_by_team(match_infos))
figs.append(plot_drakes_taken(match_infos))
figs.append(plot_drakes_wr(match_infos))
figs.append(plot_game_infos_win_correlation(match_infos))
figs.append(plot_wr_champions(game_stats))
figs.append(plot_wr_champions_by_team(game_stats))
figs.append(plot_wr_bans(match_infos))
figs.append(plot_wr_bans_by_team(match_infos))
figs += plot_average_stats_by_role(game_stats)

print(figs)