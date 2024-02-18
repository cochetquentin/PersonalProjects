from tools import *
from flask import Flask, render_template, request
from os import path

app = Flask(__name__)

tournaments_analyzed = {
    "lfl_spring_2024_rs": "https://gol.gg/tournament/tournament-matchlist/LFL%20Spring%202024/",
    "lec_winter_2024_rs": "https://gol.gg/tournament/tournament-matchlist/LEC%20Winter%20Season%202024/",
    "lec_winter_2024_po": "https://gol.gg/tournament/tournament-matchlist/LEC%20Winter%20Playoffs%202024/",
    "lpl_spring_2024_rs": "https://gol.gg/tournament/tournament-matchlist/LPL%20Spring%202024/",
    "lck_spring_2024_rs": "https://gol.gg/tournament/tournament-matchlist/LCK%20Spring%202024/",
    "lcs_spring_2024_rs": "https://gol.gg/tournament/tournament-matchlist/LCS%20Spring%202024/",
    "lcs_spring_2024_po": "https://gol.gg/tournament/tournament-matchlist/LCS%20Spring%20Playoffs%202024/",
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/analyse" , methods=["GET"])
def analyse():
    args = request.args
    tournament = args.get("tournament")

    if tournament not in tournaments_analyzed:
        return render_template('analyse.html', possible_tournaments=list(tournaments_analyzed.keys()))

    if not path.exists(f"static/imgs_to_plot/{tournament}") or args.get("refresh", "false") == "true":
        url = tournaments_analyzed[tournament]
        match_infos, game_stats = get_all_match_data_from_gol_url(url)
        create_all_plots_data(match_infos, game_stats, tournament)
    img_datas, team_names = get_all_plots_data(tournament)

    return render_template('analyse.html', img_datas=img_datas, team_names=team_names, possible_tournaments=list(tournaments_analyzed.keys()))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
