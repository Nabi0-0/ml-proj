from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")

# Convert available splits to pandas
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# test_labels might be in the test split or a separate file
# Let's check what's actually available
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")
