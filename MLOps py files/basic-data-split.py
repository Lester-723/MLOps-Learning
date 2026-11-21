import pandas as pd 
from sklearn.model_selection import train_test_split

def read_csv(path):
    df = pd.read_csv(path)

    return df
