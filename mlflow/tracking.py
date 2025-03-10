import mlflow 
import mlflow.sklearn
import logging 

# Configure Logging 
logging.basicConfig(
    filename = "logs/mlflow.log",
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s]: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

def setup_mlflow():
    """Configures MLFlow tracking server"""
    tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit Card Fraud Detection") 
    logging.info(f"MLFlow Tracking server started at {tracking_uri}")  

if __name__ == "__main__":
    setup_mlflow() 
    