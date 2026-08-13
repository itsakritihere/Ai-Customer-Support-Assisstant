import pandas as pd

df = pd.read_csv("banking77_train.csv")

print("Number of labels:", df["label"].nunique())

print("\nLabel -> Example")
print("=" * 60)

for label in sorted(df["label"].unique()):
    example = df[df["label"] == label]["text"].iloc[0]
    print(f"{label:2} -> {example}")