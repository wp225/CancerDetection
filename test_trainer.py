import mlruns
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import dagshub
dagshub.init(repo_owner='georgejoshi10', repo_name='MLflow', mlflow=True)


mlruns.set_tracking_uri('https://dagshub.com/georgejoshi10/MLflow.mlflow')
# mlflow.set_tracking_uri('http://ec2-54-237-116-22.compute-1.amazonaws.com:5000')


db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

with mlruns.start_run(run_name='test'):
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlruns.log_metric('mean_squared_error', mse)
    mlruns.log_metric('r2_score', r2)
