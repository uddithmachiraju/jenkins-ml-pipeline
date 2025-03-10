import pandas as pd 
import pickle 
import time 
import logging 
import os 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score 

os.makedirs("logs", exist_ok = True)
# Configure logging
logging.basicConfig(
    filename = "logs/training.log",
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s]: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

def train_model(data_path, save_path):
    """Loads, trains and Saves the model"""
    try:
        logging.info("Training process Started!")

        # Read the file
        df = pd.read_csv(data_path)
        features = df.drop(columns = ['Class'])
        labels = df['Class']
        logging.info("Extracted features and labels from data")

        # Split data 
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True) 
        logging.info("Data splitted into train and test sets")

        # Apply SMOTE to data
        smote = SMOTE() 
        X_smote, y_smote = smote.fit_resample(X_train, y_train) 
        logging.info("Applied SMOTE on training data.")

        # Train the model
        model = RandomForestClassifier(
            n_estimators = 100
        )
        start_time = time.time() 
        model.fit(X_smote, y_smote) 
        training_time = round(time.time() - start_time) 
        logging.info(f"Model fitted to the training data with in {training_time} seconds") 

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        logging.info(f"Model trained with accuracy: {accuracy}")

        # Save the model with timestamp
        os.makedirs(save_path, exist_ok = True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") 
        model_path = f"{save_path}/trained_model_{timestamp}.pkl" 
        with open(model_path, "wb") as file:
            pickle.dump(
                {
                    "model": model,
                    "accuracy" : accuracy
                },
                file
            )
        logging.info(f"Model saved at {model_path}")

    except Exception as e:
        logging.info(f"Error in trainig: {str(e)}") 

if __name__ == "__main__":
    train_model(
        data_path = "data/processed_data.csv",
        save_path = "models"
    ) 