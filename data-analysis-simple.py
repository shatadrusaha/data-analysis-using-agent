"""                     Load the libraries.                     """
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


"""                     User defined variables.                     """
load_dotenv()

role_promt = [
    {
        "role": "system",
        "content": ""
    }
]

# Path to the data file.
# Manually downloaded from https://www.kaggle.com/datasets/jiweiliu/nyc-parking-violations-2022-small/code
file_path = "data/nyc_parking_violations_2022_small.parquet"


"""                     Load the data.                     """
# Load the latest version
df = pd.read_parquet(path=file_path, engine="pyarrow")

print("First 5 records:", df.head())