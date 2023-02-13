from string import ascii_letters
import random
import pandas as pd


def feature_engineering(df, ab):
    return df

def data_viz(df):
    return df, df

def grid_search():
    hyperparameters = []
    return hyperparameters

def collect_dataset(Customer_Data, Towers_and_Complaints,
                                       Network_Logs, CDRs, IMEI_info, Other_Data_Sources):
    return pd.DataFrame[Customer_Data, Towers_and_Complaints,
                                       Network_Logs, CDRs, IMEI_info, Other_Data_Sources]

def random_string_generator(size=10, chars=ascii_letters):
    df={}
    df["udmi"]="".join(random.choice(chars) for _ in range(size))
    return pd.DataFrame(df,index=[0])
