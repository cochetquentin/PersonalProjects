from tools import *
from flask import Flask, render_template, request

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

    url = tournaments_analyzed[tournament]
    match_infos, game_stats = get_all_match_data_from_gol_url(url)
    img_datas = get_all_plots_data(match_infos, game_stats)

    return render_template('analyse.html', img_datas=img_datas, team_names=match_infos["Team"].unique().tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
