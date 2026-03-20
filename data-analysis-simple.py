"""
Resources:
    - https://www.youtube.com/watch?v=CCjxuVhKKok
    - https://www.kaggle.com/datasets/jiweiliu/nyc-parking-violations-2022-small
    - https://docs.api.nvidia.com/nim/reference/llm-apis
    - https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemotron-ultra-253b-v1

Data:
    - https://www.kaggle.com/datasets/jiweiliu/nyc-parking-violations-2022-small
"""
# Load the libraries.
import os
import yaml
import kagglehub
import glob
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Download the dataset from Kaggle using kagglehub.
os.makedirs(config["data"]["folder"], exist_ok=True)

path_data = kagglehub.dataset_download(
    handle=config["data"]["dataset_handle"],
    output_dir=config["data"]["folder"]
)
print(f"Path to dataset files: {path_data}\n")

# Load the parquet file into a DataFrame.
parquet_files = glob.glob(os.path.join(path_data, "*.parquet"))
if parquet_files:
    file_path = parquet_files[0]
    print(f"Found file: {file_path}")
    df = pd.read_parquet(file_path, engine="pyarrow")

# df.head()
# print("First 5 records:", df.head())

# Define the system prompt for the agent.
role_promt = [
    {
        "role": "system",
        "content": ""
    }
]

# Define the agent function to process the query and return the response.
def agent(
    query: str,
    file_path: str,
    api_endpoint: str = config["api"]["endpoint"],
    api_key: str = os.getenv("NV_API_KEY"),
    model: str = config["api"]["model"]
):
    """Agent function to process the query and return the response."""
    df = pd.read_parquet(path=file_path, engine="pyarrow")
    user_promt = f"""
"""