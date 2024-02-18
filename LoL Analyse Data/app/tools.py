import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import requests as rq
import io
import base64
from os import path, mkdir, listdir

from tqdm.auto import tqdm


def get_match_links_from_gol_url(url:str) -> list:
    """
    Get all match links from a given url
    """
    page = rq.get(url, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.find("table").findAll("a", href = True)
    return [f"https://gol.gg{link['href'][2:]}" for link in links if not "preview" in link['href']]


def get_size_of_bo(match_link:str) -> int:
    page = rq.get(match_link, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    soup = BeautifulSoup(page.content, "html.parser")
    return len(soup.findAll("ul", {"class": "navbar-nav mr-auto mt-2 mt-lg-0"})[1].findAll("li")) - 2


def get_match_links_from_bo(match_link:str, bo:int) -> list:
    initial_value = int(match_link.split("/")[-1]) 
    return [f"https://gol.gg/game/stats/{initial_value + i}" for i in range(bo)]


def get_match_data_from_match_link(link:str, game_id:int=0) -> tuple([pd.DataFrame, pd.DataFrame]):
    url = f"{link}/page-game/"
    page = rq.get(url, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    soup = BeautifulSoup(page.content, "html.parser")

    # Get game time and convert to seconds
    game_time = soup.find("div", {"class" : "col-6 text-center"}).h1.text
    game_time = game_time.split(":")
    game_time = int(game_time[0]) * 60 + int(game_time[1])

    # Get blue team name and if they won
    blue_team_name = soup.find("div", {"class" : "col-12 blue-line-header"}).text.strip()
    blue_team_name, blue_side_win = blue_team_name.split(" - ")
    blue_side_win = int(blue_side_win == "WIN")

    # Get red team name
    red_team_name = soup.find("div", {"class" : "col-12 red-line-header"}).text.strip()
    red_team_name, red_side_win = red_team_name.split(" - ")

    # Get stats (kills, towers, dragons, barons, golds)
    blue_stats = [stat.text.strip() for stat in soup.findAll("span", {"class" : "score-box blue_line"})]
    red_stats = [stat.text.strip() for stat in soup.findAll("span", {"class" : "score-box red_line"})]

    blue_kills = int(blue_stats[0])
    blue_towers = int(blue_stats[1])
    blue_dragons = int(blue_stats[2])
    blue_barons = int(blue_stats[3])
    blue_golds = float(blue_stats[4].replace("k", ""))

    red_kills = int(red_stats[0])
    red_towers = int(red_stats[1])
    red_dragons = int(red_stats[2])
    red_barons = int(red_stats[3])
    red_golds = float(red_stats[4].replace("k", ""))

    stats = soup.findAll("div", {"class": "col-2"})
    # Check if blue team got first blood, for that I check if there are 2 images in the first column
    blue_first_blood = int(len(stats[0].findAll("img")) == 2)
    # Check if blue team got first tower, for that I check if there are 2 images in the second column
    blue_first_tower = int(len(stats[1].findAll("img")) == 2)

    # Take which drakes were taken by each team
    blue_drakes = [drake["alt"] for drake in stats[2].findAll("img")[1:]]
    red_drakes = [drake["alt"] for drake in stats[10].findAll("img")[1:]]

    # Get draft (bans and picks) for each team
    draft = soup.findAll("div", {"class": "col-10"})
    blue_bans = [ban["alt"] for ban in draft[0].findAll("img")]
    blue_picks = [pick["alt"] for pick in draft[1].findAll("img")]
    red_bans = [ban["alt"] for ban in draft[2].findAll("img")]
    red_picks = [pick["alt"] for pick in draft[3].findAll("img")]

    match_infos_part = [
        [game_id, game_time, blue_team_name, blue_side_win, blue_kills, blue_towers, blue_dragons, blue_barons, blue_golds, blue_first_blood, blue_first_tower, blue_drakes, blue_bans, blue_picks, "Blue"],
        [game_id, game_time, red_team_name, 1-blue_side_win, red_kills, red_towers, red_dragons, red_barons, red_golds, blue_first_blood, blue_first_tower, red_drakes, red_bans, red_picks, "Red"]
    ]
    match_infos_part = pd.DataFrame(match_infos_part, columns = ["Game ID", "Game Time", "Team", "Win", "Kills", "Towers", "Dragons", "Barons", "Golds", "First Blood", "First Tower", "Drakes", "Bans", "Picks", "Side"])

    # Get full stats page
    url = f"{link}/page-fullstats/"
    page = rq.get(url, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    soup = BeautifulSoup(page.content, "html.parser")

    # Get data from the game for each player
    game_stats_part = {}
    for stat in soup.find("table").findAll("tr")[1:]:
        tds = stat.findAll("td")
        game_stats_part[tds[0].text.strip()] = [elem.text.strip() for elem in tds[1:]]

    game_stats_part = pd.DataFrame.from_dict(game_stats_part)
    game_stats_part.drop(columns=["CS in Enemy Jungle", "Solo kills", "KDA", "GOLD%", "VS%", "DMG%", "KP%"], inplace=True)
    game_stats_part.replace("", "NaN", inplace=True)
    game_stats_part = game_stats_part.infer_objects()
    game_stats_part[game_stats_part.columns[2:]] = game_stats_part[game_stats_part.columns[2:]].astype(float)

    # Add the an Id for the game, the team name and if they won
    game_stats_part["Game ID"] = game_id
    game_stats_part["Team"] = [blue_team_name]*5 + [red_team_name]*5
    game_stats_part["Win"] = [blue_side_win]*5 + [1 - blue_side_win]*5
    game_stats_part["Champion"] = blue_picks + red_picks

    return match_infos_part, game_stats_part


def get_all_match_data_from_gol_url(url:str) -> tuple([pd.DataFrame, pd.DataFrame]):
    match_links = get_match_links_from_gol_url(url)
    bos = [get_size_of_bo(link) for link in tqdm(match_links)]
    match_links = ["/".join(match.split("/")[:-2]) for match in match_links]
    match_links = [get_match_links_from_bo(link, bo) for link, bo in zip(match_links, bos)]
    match_links = [link for sublist in match_links for link in sublist]

    match_infos =  []
    game_stats = []
    for game_id, link in tqdm(enumerate(match_links), total = len(match_links)):
        match_info, game_stat = get_match_data_from_match_link(link, game_id)
        match_infos.append(match_info)
        game_stats.append(game_stat)

    match_infos = pd.concat(match_infos).reset_index(drop=True)
    game_stats = pd.concat(game_stats).reset_index(drop=True)

    return match_infos, game_stats


def plot_to_base64(fig:plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def save_plot(fig:plt.Figure, path:str) -> None:
    """
    Save a matplotlib figure to a file
    """
    fig.savefig(path, format="png", dpi=100, bbox_inches="tight")


def plot_side_wr(match_infos:pd.DataFrame) -> plt.Figure:
    """
    Plot the win rate for each side
    """
    fig, ax = plt.subplots(1, 1)
    match_infos[["Side", "Win"]].groupby("Side").mean().plot.pie(
        y = "Win",
        autopct = "%.2f%%",
        figsize = (3, 3),
        title = "Win rate by side",
        legend = False,
        colors = ["tab:blue", "tab:red"],
        ylabel = "",
        ax = ax
    )
    return fig
    


def plot_side_wr_by_team(match_infos:pd.DataFrame) -> plt.Figure:
    """
    Plot the win rate for each team by side
    """
    fig, ax = plt.subplots(1, 1)
    match_infos[["Side", "Team", "Win"]].groupby(["Team", "Side"]).mean().unstack().plot.bar(
        y = "Win",
        figsize = (10, 3),
        title = "Win rate by team and side",
        color = ["tab:blue", "tab:red"],
        ylabel = "",
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_average_stats_by_team(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos[["Team", "Kills", "Towers", "Dragons", "Barons"]].groupby("Team").mean().round(1).plot.bar(
        figsize = (15, 5),
        title = "Average stats by team",
        ylabel = "",
        ax = ax
    )

    for p in fig.patches:
        fig.annotate(str(float(p.get_height())), (p.get_x(), p.get_height()+0.1))

    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_drakes_taken_by_team(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos[["Team", "Drakes"]].explode("Drakes").groupby(["Team", "Drakes"]).size().unstack().plot.bar(
        figsize = (15, 5),
        title = "Drakes taken by team",
        ylabel = "",
        color=["tab:olive", "tab:grey", "black", "tab:cyan", "tab:red", "tab:brown", "tab:blue"],
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_drakes_taken(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos["Drakes"].explode().value_counts().sort_index().plot.pie(
        autopct = "%.2f%%",
        figsize = (5, 5),
        title = "Drakes taken",
        ylabel = "",
        colors=["tab:olive", "tab:grey", "white", "tab:cyan", "tab:red", "tab:brown", "tab:blue"],
        ax = ax
    )
    return fig


def plot_drakes_wr(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos[["Drakes", "Win"]].explode("Drakes").groupby("Drakes").mean().sort_values("Win").plot.barh(
        figsize = (5, 5),
        title = "Win rate by drake",
        ylabel = "",
        color = "tab:blue",
        legend = False,
        ax = ax
    )
    plt.grid(axis = "x", linestyle = "--")
    return fig


def plot_game_infos_win_correlation(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos[["Win", "Kills", "Towers", "Dragons", "Barons"]].corr()["Win"].iloc[1:].sort_values().plot.barh(
        figsize = (8, 3),
        title = "Correlation between stats and win",
        ylabel = "",
        ax = ax
    )
    return fig


def plot_game_first_blood_turret(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    match_infos[["Team", "First Blood", "First Tower"]].groupby("Team").mean().plot.bar(
        figsize = (10, 3),
        title = "First blood and first tower rate",
        ylabel = "",
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    return fig


def plot_game_stats_win_correlation(game_stats:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1) 
    game_stats.drop(columns=["Player", "Role", "Team", "Champion", "Game ID"]).corr()["Win"].iloc[:-1].sort_values().plot.barh(
        figsize = (8, 10),
        title = "Correlation between stats and win",
        ylabel = "",
        ax = ax
    )
    plt.grid(axis = "x", linestyle = "--")
    return fig


def plot_wr_champions(game_stats:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1) 
    champ_wr = game_stats[["Win", "Champion"]].groupby("Champion").agg(["mean", "count"])["Win"].sort_values(["mean", "count"], ascending=False)
    champ_wr.rename(columns={"mean": "Win rate", "count": "Games played"}, inplace=True)
    champ_wr[champ_wr["Games played"] > 5].plot.bar(
        figsize = (15, 3),
        title = "Win rate by champion (At least 5 games played)",
        secondary_y = "Games played",
        ax = ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(axis = "y", linestyle = "--")
    return fig


def plot_wr_champions_for_team(game_stats:pd.DataFrame, team:str) -> plt.Figure:
    fig, ax = plt.subplots(1, 1) 
    team_champ_wr = game_stats[game_stats["Team"] == team][["Win", "Champion"]].groupby("Champion").agg(["mean", "count"])["Win"].sort_values(["mean", "count"], ascending=False)
    team_champ_wr.rename(columns={"mean": "Win rate", "count": "Games played"}, inplace=True)
    team_champ_wr.plot.bar(
        figsize = (15, 3),
        title = f"Win rate by champion for {team}",
        secondary_y = "Games played",
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_wr_champions_by_team(game_stats:pd.DataFrame) -> list([plt.Figure]):
    figs = []
    for team in game_stats["Team"].unique():
        figs.append({
            "fig": plot_wr_champions_for_team(game_stats, team),
            "name": team
        })
    return figs


def plot_wr_bans(match_infos:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1) 
    ban_wr = match_infos[["Bans", "Win"]].explode("Bans").groupby("Bans").agg(["mean", "count"])["Win"].sort_values(["mean", "count"], ascending=False)
    ban_wr.rename(columns={"mean": "Win rate", "count": "Number of bans"}, inplace=True)
    ban_wr[ban_wr["Number of bans"] > 7].plot.bar(
        figsize = (15, 3),
        title = "Win rate by ban (At least 7 bans)",
        secondary_y = "Number of bans",
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_wr_bans_for_team(match_infos:pd.DataFrame, team:str) -> plt.Figure:
    fig, ax = plt.subplots(1, 1) 
    team_ban_wr = match_infos[match_infos["Team"] == team][["Bans", "Win"]].explode("Bans").groupby("Bans").agg(["mean", "count"])["Win"].sort_values(["mean", "count"], ascending=False)
    team_ban_wr.rename(columns={"mean": "Win rate", "count": "Number of bans"}, inplace=True)
    team_ban_wr.plot.bar(
        figsize = (15, 3),
        title = f"Win rate by ban for {team}",
        secondary_y = "Number of bans",
        ax = ax
    )
    plt.grid(axis = "y", linestyle = "--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def plot_wr_bans_by_team(match_infos:pd.DataFrame) -> list([plt.Figure]):
    figs = []
    for team in match_infos["Team"].unique():
        figs.append({
            "fig": plot_wr_bans_for_team(match_infos, team),
            "name": team
        })
    return figs


def plot_average_stats_for_role(game_stats:pd.DataFrame, role:str) -> plt.Figure:
    stats_to_plots = ["Level", "Kills", "Deaths", "Assists", "CSM", "GPM", "Vision Score", "Wards placed", "Wards destroyed", "Total damage to Champion", "DPM", "GD@15", "CSD@15", "XPD@15", "LVLD@15"]
    
    fig, axis = plt.subplots(5, 3, figsize = (20, 20))
    fig.suptitle(f"Average stats for {role}", fontsize = 20, y = 0.93)
    for ax, stat in zip(axis.flatten(), stats_to_plots):
        game_stats[game_stats["Role"] == role].groupby("Player")[stats_to_plots].mean().sort_values(stat, ascending=False)[stat][::-1].plot.barh(
            ax = ax, 
            title = stat,
            ylabel = "",
        )
        ax.grid(axis = "x", linestyle = "--")
    return fig



def plot_average_stats_by_role(game_stats:pd.DataFrame) -> list([plt.Figure]):
    figs = []
    for role in game_stats["Role"].unique():
        figs.append({
            "fig": plot_average_stats_for_role(game_stats, role),
            "name": role
        })
    return figs
        


def create_all_plots_data(match_infos:pd.DataFrame, game_stats:pd.DataFrame, tournament:str) -> None:
    folder = f"static/imgs_to_plot/{tournament}/"

    if not path.exists(folder):
        mkdir(folder)
    
    #Create plots
    fig = plot_side_wr(match_infos)
    save_plot(fig, f"{folder}plot_side_wr__side_wr.png")
    plt.close(fig)
    

    fig = plot_side_wr_by_team(match_infos)
    save_plot(fig, f"{folder}plot_side_wr_by_team__side_wr.png")
    plt.close(fig)

    fig = plot_average_stats_by_team(match_infos)
    save_plot(fig, f"{folder}plot_average_stats_by_team__objectives_plot.png")
    plt.close(fig)

    if len(match_infos["Drakes"].explode("Drakes").value_counts()) > 1:
        fig = plot_drakes_taken_by_team(match_infos)
        save_plot(fig, f"{folder}plot_drakes_taken_by_team__objectives_plot.png")
        plt.close(fig)

        fig = plot_drakes_taken(match_infos)
        save_plot(fig, f"{folder}plot_drakes_taken__objectives_plot.png")
        plt.close(fig)

        fig = plot_drakes_wr(match_infos)
        save_plot(fig, f"{folder}plot_drakes_wr__objectives_plot.png")
        plt.close(fig)

    fig = plot_game_infos_win_correlation(match_infos)
    save_plot(fig, f"{folder}plot_game_infos_win_correlation__objectives_plot.png")
    plt.close(fig)

    fig = plot_game_first_blood_turret(match_infos)
    save_plot(fig, f"{folder}plot_first_blood_turret__objectives_plot.png")
    plt.close(fig)

    fig = plot_game_stats_win_correlation(game_stats)
    save_plot(fig, f"{folder}plot_game_stats_win_correlation__objectives_plot.png")
    plt.close(fig)

    fig = plot_wr_champions(game_stats)
    save_plot(fig, f"{folder}plot_wr_champions__champions_wr.png")
    plt.close(fig)

    figs = plot_wr_champions_by_team(game_stats)
    for fig in figs:
        save_plot(fig.get("fig"), f"{folder}plot_wr_champions_{fig.get('name')}__{fig.get('name')} champions_wr by_team.png")
        plt.close(fig.get("fig"))

    fig = plot_wr_bans(match_infos)
    save_plot(fig, f"{folder}plot_wr_bans__bans_wr.png")
    plt.close(fig)


    figs = plot_wr_bans_by_team(match_infos)
    for fig in figs:
        save_plot(fig.get("fig"), f"{folder}plot_wr_bans_{fig.get('name')}__{fig.get('name')} bans_wr by_team.png")
        plt.close(fig.get("fig"))

    figs = plot_average_stats_by_role(game_stats)
    for fig in figs:
        save_plot(fig.get("fig"), f"{folder}plot_average_stats_by_role_{fig.get('name')}__{fig.get('name')} role_stats.png")
        plt.close(fig.get("fig"))

    with open(f"{folder}team_name", "w") as file:
        file.write("\n".join(match_infos["Team"].unique()))


def get_all_plots_data(tournament:str) -> tuple([list([dict]), list([str])]):
    folder = f"imgs_to_plot/{tournament}/"
    files = listdir("static/"+folder)
    
    path_class = [{"path": folder+f, "class": f.split("__")[1].split(".")[0]} for f in files if f.endswith(".png")]
    
    with open(f"static/{folder}team_name", "r") as file:
        team_names = file.read().split("\n")

    return path_class, team_names