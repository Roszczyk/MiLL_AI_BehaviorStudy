import pandas as pd
from pathlib import Path

df = pd.read_csv(f"{Path(__file__).parent}/prepared_data.csv")

df["people_count"] = df['people_count'].where(df["people_count"]>0, 0.0)
df["people_count"] = df['people_count'].where(df["people_count"]==0, 1.0)
df = df.drop(columns=['Unnamed: 0'])
df["is_present"] = df["people_count"]
df = df.drop(columns=['people_count'])
df = df[['is_present'] + [col for col in df.columns if col != 'is_present']]

print(df)
df.to_csv(Path(__file__).parent / "prepared_data_for_classifier.csv")