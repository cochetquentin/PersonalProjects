from tools import *
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/lec_winter_2024", methods=["GET"])
def lec_winter_2024():
    # Load the data
    url = "https://gol.gg/tournament/tournament-matchlist/LEC%20Winter%20Season%202024/"
    match_infos, game_stats = get_all_match_data_from_gol_url(url)
    img_datas = get_all_plots_data(match_infos, game_stats)

    return render_template('index.html', img_datas=img_datas)


@app.route("/lfl_spring_2024", methods=["GET"])
def lfl_spring_2024():
    # Load the data
    url = "https://gol.gg/tournament/tournament-matchlist/LFL%20Spring%202024/"
    match_infos, game_stats = get_all_match_data_from_gol_url(url)
    img_datas = get_all_plots_data(match_infos, game_stats)

    return render_template('index.html', img_datas=img_datas)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
