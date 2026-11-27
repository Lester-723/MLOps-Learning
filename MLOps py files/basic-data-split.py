import pandas as pd 
import argparse as args
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


parameter_parser = args.ArgumentParser(description= "Python Arguments for ML model preparation")

parameter_parser.add_argument(
    "--input_data",
    type = str,
    help= "Add the file path from your folder",
    required= True
)

parameter_parser.add_argument(
    "--test_size",
    type = float,
    help = "Enter the value in float data type to define what percent of data should go for testing",
    default = 0.2
)

parameter_parser.add_argument(
    "--random_state", 
    type = int,
    help = "Enter the seed value for the pattern you want your data to follow while being randomised",
    default= 50
)

para_args = parameter_parser.parse_args()




def read_file(para_args):
    try : 
        df = pd.read_csv(para_args.input_data)
        return df
        print(f"The Dataframe has been processed successfully") 

    except Exception as e:
        print("An error has occurred while reading the file")
        print("None")

def return_dtype_columns(para_args): 
    df = read_file(para_args)
    object_dtype = df.select_dtypes("object").columns
    numeric_dtype = df.select_dtypes("int").columns
    float_dtype = df.select_dtypes("float").columns

    if type(df) is None:
        return None

    return object_dtype , numeric_dtype , float_dtype

def return_train_test_split(para_args):
    df = read_file(para_args)

    X_features = ['job', 'default', 'housing', 'loan', 'month', 'age', 'balance', 'duration', 'campaign']
    Y_features = ['Subscribed']

    X_train , X_temp , Y_train , Y_temp = train_test_split(df[X_features], df[Y_features] ,test_size = para_args.test_size  ,random_state = para_args.random_state )

    return (f"X Training Data : {X_train.shape}, X Temp Data : {X_temp.shape} , Y Training Data : {Y_train.shape} , Y Temp Data : {Y_temp.shape}")

if __name__ == "__main__":

    read_result = read_file(para_args)
    column_result = return_dtype_columns(para_args)
    train_test_split = return_train_test_split(para_args)

    print(read_result)
    print(column_result)
    print(train_test_split)







