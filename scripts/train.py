import pandas as pd 
import pickle 
import time 
import logging 
import os 
import mlflow 
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score 

os.makedirs("logs", exist_ok = True)
# Configure logging
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train_model(data_path, save_path, max_evals=10):
    """Trains and saves the best model using Hyperopt tuning and MLflow tracking."""
    try:
        logging.info("Starting model training process...")

        # Load dataset
        df = pd.read_csv(data_path)
        features = df.drop(columns=["Class"])
        labels = df["Class"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info("Data preprocessing completed.")

        # Define hyperparameter search space
        search_space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
            'max_depth': hp.choice('max_depth', [5, 10, 15, 20]),
            'min_samples_split': hp.uniform('min_samples_split', 0.1, 0.5)
        }

        # Define objective function for optimization
        def objective(params):
            with mlflow.start_run(nested=True):
                logging.info(f"Training model with params: {params}")

                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    random_state=42
                )

                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = round(time.time() - start_time)

                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)
                logging.info(f"Model trained in {training_time} sec, Accuracy: {accuracy}")

                # Minimize negative accuracy for Hyperopt
                return -accuracy

        # Run Hyperparameter Optimization
        mlflow.set_experiment("RandomForest Hyperopt Tuning")
        trials = Trials()
        best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        logging.info(f"Best Hyperparameters: {best_params}")

        # Train final model with best hyperparameters
        with mlflow.start_run():
            best_model = RandomForestClassifier(
                n_estimators=[50, 100, 150, 200][best_params['n_estimators']],
                max_depth=[5, 10, 15, 20][best_params['max_depth']],
                min_samples_split=best_params['min_samples_split'],
                random_state=42
            )
            best_model.fit(X_train, y_train)

            # Final evaluation
            y_pred = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("final_accuracy", final_accuracy)

            # Save best model with versioning
            os.makedirs(save_path, exist_ok=True)
            model_path = f"{save_path}/best_model_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
            with open(model_path, "wb") as file:
                pickle.dump(
                {
                    "model": best_model,
                    "accuracy" : final_accuracy, 
                    "model-parameters" : best_model.get_params() 
                }, 
                file
            )

            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.log_artifact(model_path) 
            logging.info(f"Best Model saved at {model_path} with accuracy {final_accuracy}")

    except Exception as e:
        logging.error(f"Error in training: {str(e)}")

# Run training if script is executed
if __name__ == "__main__":
    os.makedirs("models", exist_ok = True)
    train_model(
        data_path="data/processed_data.csv",
        save_path="models",
        max_evals=10  # Number of hyperparameter tuning iterations
    )
