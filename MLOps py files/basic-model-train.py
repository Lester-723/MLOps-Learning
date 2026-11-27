import pandas as pd 
import argparse as args
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , f1_score , confusion_matrix, PrecisionRecallDisplay
import mlflow

parameter_parser = args.ArgumentParser(description= "Python Arguments for ML model preparation")

parameter_parser.add_argument(
    "--azure_tracking_id",
    type = str,
    help = "Add the tracking id to your Azure ML studio workspace",
    required= True
)

parameter_parser.add_argument(
    "--azure_experiment_name",
    type = str,
    help = "Specify the experiment where you want you mlflow job to run",
    required = True
)

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

parameter_parser.add_argument(
    "--validation_size",
    type = int,
    help = "Enter the validation size for splitting the temp data into validation and test data",
    default = 0.4
)

para_args = parameter_parser.parse_args()


def read_file(para_args):
    try : 
        df = pd.read_csv(para_args.input_data)
        df = df.drop(['index'], axis = 1)
        return df
        print(f"The Dataframe has been processed successfully") 

    except Exception as e:
        print("An error has occurred while reading the file")

def return_key_columns(para_args): 
    df = read_file(para_args)
    object_cols = df.select_dtypes("str").columns
    numeric_cols = df.select_dtypes("int").columns
    float_cols = df.select_dtypes("float").columns

    return list(object_cols) , list(numeric_cols) , list(float_cols)

def return_transformation(para_args):
    df = read_file(para_args)
    object_cols , numeric_cols , float_cols = return_key_columns(para_args)
    new_df = df.copy()
    
    for feature in object_cols:
        feature_values = new_df[feature].value_counts()
        transformed_feature = pd.factorize(feature_values)[0]

        for old_feature_name , new_feature_name in zip(new_df[feature].unique(), transformed_feature):
                new_df.replace({
                old_feature_name : new_feature_name
                 }, inplace= True)
            
    combined_cols = object_cols + numeric_cols + float_cols   
    X_data = new_df[combined_cols]
    X_data = new_df.drop(['Subscribed'],axis = 1)
    Y_data = new_df['Subscribed']

    return X_data , Y_data

def return_train_test_split(para_args):
    
    X_data , Y_data = return_transformation(para_args)


    X_train , X_temp , Y_train , Y_temp = train_test_split(X_data , Y_data ,test_size = para_args.test_size  ,random_state = para_args.random_state)

    X_val , X_test , Y_val , Y_test = train_test_split(X_temp, Y_temp , test_size = para_args.validation_size , random_state = para_args.random_state)

    # Recording the structure of the Data split

    print(f"X Train :{X_train.shape}")
    print(f"X Temp : {X_temp.shape}")
    print(f"X Val : {X_val.shape}")
    print(f"X Test : {X_test.shape}")

    return X_train , X_temp , X_val , X_test , Y_train , Y_temp , Y_val , Y_test


def model_training_logging():
    X_train , X_temp , X_val , X_test , Y_train , Y_temp , Y_val , Y_test = return_train_test_split(para_args)

    mlflow.set_tracking_id(para_args.azure_tracking_id)

    mlflow.set_experiment(experiment_name = para_args.azure_experiment_name)

    try : 
        with mlflow.start_run(run_name= 'Basic model training log', description= 'This mlflow is through a command job from a script'):
            model = LogisticRegression()
            model.fit(X_train,Y_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            y_val_score = model.predict_proba(X_val)[:,1]


            mlflow.log_params(
                {
                    "Model used" : "Logistic Regression",
                    "Validation Accuracy" : accuracy_score(Y_train , y_val_pred),
                    "Test Accuracy" : accuracy_score(Y_train , y_test_pred),
                
                } 
            )  

            fig = plt.fig
            PrecisionRecallDisplay.from_predictions(y_val, y_val_score , ax = plt.gca())

            mlflow.log_figure(fig , "Precision Recall figure.png")
        
    except Exception as e:
        print(f"An Error has occured while processing mlflow {e}")


if __name__ == "__main__":

    read_result = read_file(para_args)
    data_transformation = return_transformation(para_args)
    column_result = return_key_columns(para_args)
    train_test_split = return_train_test_split(para_args)
    logging_result = model_training_logging(para_args)


    print(read_result)
    print(column_result)
    print(data_transformation)
    print(train_test_split)
    print(logging_result)







