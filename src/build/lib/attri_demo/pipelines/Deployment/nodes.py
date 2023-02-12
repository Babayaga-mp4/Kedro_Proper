
from string import ascii_letters
import random
import pandas as pd


def random_string_generator(size=10, chars=ascii_letters):
    df={}
    df["udmi"]="".join(random.choice(chars) for _ in range(size))
    return pd.DataFrame(df,index=[0])
