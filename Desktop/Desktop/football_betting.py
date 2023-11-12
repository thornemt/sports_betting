import numpy as np
import pandas as pd
import glob
import os
import math
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
from collections import defaultdict


GOAL_FEATURES = ["HTeam",
           "ATeam",
           "FTHG",
           "FTAG",
           "FTR",
            ]


PATH = os.getcwd() + "/Football_Data"

HOME_MODEL = PATH + "/home_model_7_11_23.sav"
AWAY_MODEL = PATH + "/away_model_7_11_23.sav"


def get_models():
    home_model = pickle.load(open(HOME_MODEL, 'rb'))
    away_model = pickle.load(open(AWAY_MODEL, 'rb'))
    return home_model, away_model


def get_data():
    files = glob.glob(PATH + "/EPL_Data*.csv")
    df = get_football_data(files)
    df = create_features(df)
    return df


def get_football_data(files):
    df = pd.concat((pd.read_csv(filename) for filename in files))
    df["Date"] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df = clean_data(df)
    return df


def clean_data(df):
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True, ascending=False)
    df.rename(columns={"HomeTeam": "HTeam", "AwayTeam": "ATeam"}, inplace=True)
    df = df[GOAL_FEATURES]
    return df


def create_features(df):
    for location in ["H", "A"]:
        df = add_rolling_mean_goals_scored(df, location)
    df = add_rolling_mean_goals_conceded(df)
    df = one_hot_encode_teams(df)
    return df


def add_rolling_mean_goals_scored(df, location: str):
    df_rev = df.iloc[::-1]
    df_rev[f"rolling_mean_{location}_scored"] = df_rev.groupby(f'{location}Team')[f'FT{location}G'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    df[f"rolling_mean_{location}_scored"] = df_rev[f"rolling_mean_{location}_scored"].iloc[::-1]
    return df


def add_rolling_mean_goals_conceded(df):
    df_rev = df.iloc[::-1]
    df_rev["rolling_mean_H_conceded"] = df_rev.groupby('HTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    df_rev["rolling_mean_A_conceded"] = df_rev.groupby('ATeam')['FTHG'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    df["rolling_mean_H_conceded"] = df_rev["rolling_mean_H_conceded"].iloc[::-1]
    df["rolling_mean_A_conceded"] = df_rev["rolling_mean_A_conceded"].iloc[::-1]
    return df


def one_hot_encode_teams(df):
    H_teams = np.unique(df["HTeam"])
    for team in H_teams:
        df[f"{team}H"] = np.where(df["HTeam"] == team, 1, 0)
        df[f"{team}A"] = np.where(df["ATeam"] == team, 1, 0)
    return df


def predict_goals(df, home_model, away_model):
    Hpred = home_model.predict(df)
    Apred = away_model.predict(df)
    res_df = pd.DataFrame(data={"H_Pred": Hpred, "A_Pred": Apred})
    return res_df
    

def predict_goal_probs(df):   
    probs = defaultdict(list)

    for team in ["H", "A"]:
        for i, l in enumerate(df[f"{team}_Pred"]):
            for j in range(11):
                probs[f"game_{i}_{team}"].append(poisson_probability(l, j))
    return probs


def poisson_probability(l, x):
    probability = ((l**x) * math.exp(-l)) / math.factorial(x)
    return probability*100


def calc_result_matrix(game_num, probs):
    
    result_matrix = np.zeros((11,11))
    
    for i in range(11):
        for j in range(11):
            result_matrix[i][j] = (probs[f"game_{game_num}_H"][i] * probs[f"game_{game_num}_A"][j]) / 10000

    return result_matrix


def rho_correction(x, y, lambda_x, mu_y, rho):
    if x==0 and y==0:
        return 1- (lambda_x * mu_y * rho)
    elif x==0 and y==1:
        return 1 + (lambda_x * rho)
    elif x==1 and y==0:
        return 1 + (mu_y * rho)
    elif x==1 and y==1:
        return 1 - rho
    else:
        return 1.0


def visualise_result_matrix(matrix):
    fig, ax = plt.subplots()
    
    ax.imshow(matrix)
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center') 

    
def generate_odds_predictions(games_to_predict, probs):
    odds =  pd.DataFrame(np.zeros((games_to_predict,9)), 
                         columns=["BTTS", "Under_1.5", "Under_2.5", "Under_3.5", "Under_4.5",
                                  "Over_1.5", "Over_2.5", "Over_3.5", "Over_4.5"],
                         index=[f"Game_{i}" for i in range(1, games_to_predict + 1)])

    for i in range(games_to_predict):
        res_matrix = calc_result_matrix(i, probs)
        odds.loc[f"Game_{i+1}", "BTTS"] = odds_both_teams_to_score(res_matrix)

        for x in [1.5, 2.5, 3.5, 4.5]:
            odds.loc[f"Game_{i+1}", f"Under_{x}"] = (1 / prob_under_x_goals(x, res_matrix))
            odds.loc[f"Game_{i+1}", f"Over_{x}"] = (1 / (1 - prob_under_x_goals(x, res_matrix)))

        for team in ["H", "A"]:
            odds.loc[f"Game_{i+1}", f"{team}_Win"] = (1 / prob_win_or_draw(res_matrix, team, "Win"))
        
        odds.loc[f"Game_{i+1}", "Draw"] = (1 / prob_win_or_draw(res_matrix, result="Draw"))
    return odds


def odds_both_teams_to_score(matrix):
    BTTS = 1 / (1 - matrix.sum(axis=1)[0] - matrix.sum(axis=0)[0] + matrix[0][0])
    return BTTS


def prob_under_x_goals(x, matrix):
    prob = 0
    for i in range(11):
        for j in range(11):
            if i + j < x:
                prob += matrix[i,j]
    return prob


def prob_win_or_draw(matrix, team="H", result="Win"):
    if result == "Win":
        prob = prob_win(team, matrix)
    elif result == "Draw":
        prob = prob_draw(matrix)
    else:
        return
    return prob


def prob_win(team, matrix):
    prob = 0
    for i in range(11):
        for j in range(11):
            if team == "H" and i > j:
                prob += matrix[i,j]
            elif team == "A" and i < j:
                prob += matrix[i,j]
    return prob


def prob_draw(matrix):
    prob = 0
    for i in range(11):
        for j in range(11):
            if i == j:
                prob += matrix[i,j]
    return prob


if __name__ == "__main__":
    games_to_predict = 10 

    home_model, away_model = get_models()
    df = get_data()
    target_df = df.drop(columns=["HTeam", "ATeam", "FTAG", "FTHG", "FTR"])

    pred_df = target_df[:games_to_predict]

    goal_df = predict_goals(pred_df, home_model, away_model)
    probs = predict_goal_probs(goal_df)

    pred_odds = generate_odds_predictions(games_to_predict, probs)
    pred_odds.insert(0, "Home", np.array(df.HTeam[:games_to_predict]))
    pred_odds.insert(1, "Away", np.array(df.ATeam[:games_to_predict]))
    print(pred_odds)
