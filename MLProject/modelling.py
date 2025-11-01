import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2, help="Test data fraction")
args = parser.parse_args()

df = pd.read_csv("california-housing_preprocessing.csv")

if "HouseAge_bin" in df.columns:
    df = df.drop(columns=["HouseAge_bin"])

target_col = "target"

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

with mlflow.start_run(run_name="basic_model_california"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    mlflow.sklearn.save_model(model, "MLProject/model")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Training selesai")
