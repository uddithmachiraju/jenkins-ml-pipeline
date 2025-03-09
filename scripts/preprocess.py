import pandas as pd 
import logging 
import os 

# Configure Logging
logging.basicConfig(
    filename = "logs/preprocess.log", 
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s]: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S", 
)

def preprocess_data(input_path, output_path):
    """Loads, Cleans, and Saves the preprocessed data"""
    try:
        logging.info("Preprocessing Started!")

        # Read the data
        df = pd.read_csv(input_path)
        logging.info(f"Loaded raw data from {input_path}") 

        # Removing duplicates
        df.drop_duplicates(inplace = True)
        logging.info("Removed Duplicates") 

        # Handling missing values
        df.fillna(df.median(), inplace = True)
        logging.info("Handled Missing values")

        # Save Processed data
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        df.to_csv(output_path, index = False)
        logging.info(f"Preprocessed file saved at {output_path}") 

    except Exception as e:
        logging.error(f"Error in preprocessing data {str(e)}") 

if __name__ == "__main__":
    preprocess_data(
        input_path = "data/creditcard.csv",
        output_path = "data/processed_data.csv" 
    ) 