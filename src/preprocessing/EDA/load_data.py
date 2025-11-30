# load_data.py

''' 
This script downloads the Harvard cold cases dataset from Hugging Face Datasets
link to dataset: https://huggingface.co/datasets/harvard-lil/cold-cases
The dataset is very large so we will filter only for Supreme Court cases
and save the filtered dataset locally as parquet and csv files for further analysis
'''

# import libraries
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# name to filter the cases by
SCOTUS_NAME = "Supreme Court of the United States"

#  Stream the dataset instead of downloading all shards into RAM
stream = load_dataset(
    "harvard-lil/cold-cases",
    split="train", # only a 'train' split available, therefore the data loaded here represents the entire dataset
    streaming=True
)

# Collect ONLY Supreme Court rows 
scotus_rows = []

for row in tqdm(stream):
    if row.get("court_full_name") == SCOTUS_NAME:
        scotus_rows.append(row)

print("Number of SCOTUS rows collected:", len(scotus_rows))

# Turn that list into a pandas DataFrame (this stays small enough to fit memory)
df = pd.DataFrame(scotus_rows)

# Save data frame in csv and parquet formats
df.to_parquet("/data/working/scotus_raw.parquet", index=False)
df.to_csv("/data/working/scotus_raw.csv", index=False)

