import os 
import pickle 
import logging 
import shutil 

os.makedirs("logs", exist_ok = True)
# Configure logging
logging.basicConfig(
    filename = "logs/save_model.log",
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s]: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

def select_best_model(path):
    """Saves the best model based on accuracy"""
    try:
        logging.info(f"Checking best model in {path}") 
        best_accuracy = 0  
        file = "" 

        for model_file in os.listdir(path):
            with open(f"models/{model_file}", "rb") as f:
                data = pickle.load(f) 
                if model_file != "best_model.pkl":
                    if data["accuracy"] > best_accuracy:
                        best_accuracy = data["accuracy"] 
                        file = model_file 
                        logging.info(f"Found best model {model_file}") 

        shutil.copy(f"models/{file}", "models/best_model.pkl")
        logging.info("Best model was saved as 'best_model.pkl' in 'models/' ")
    
    except Exception as e:
        logging.info(f"Error in saving model {str(e)}") 
    
if __name__ == "__main__":
    os.makedirs("models", exist_ok = True)
    select_best_model(path = "models/")