import pandas as pd

TRAIN_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fpr%2F6/data/train-00000-of-00001.parquet"
TEST_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fpr%2F6/data/test-00000-of-00001.parquet"

print("Downloading training dataset...")
train_df = pd.read_parquet(TRAIN_URL)

print("Downloading testing dataset...")
test_df = pd.read_parquet(TEST_URL)

print("\nDataset downloaded successfully!")

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nFirst 5 training examples:")
print(train_df.head())

print("\nNumber of categories:", train_df["label"].nunique())

print("\nLabel examples:")
print(train_df["label"].unique()[:10])

# Save locally
train_df.to_csv("banking77_train.csv", index=False)
test_df.to_csv("banking77_test.csv", index=False)

print("\nFiles saved:")
print("banking77_train.csv")
print("banking77_test.csv")