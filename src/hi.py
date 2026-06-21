'''import pandas as pd

df = pd.read_csv("data/cleaned_events.csv")

print("\nEVENT TYPES")
print(df["event_type"].unique())

print("\nEVENT CAUSES")
print(df["event_cause"].unique())

print("\nPRIORITIES")
print(df["priority"].unique())

print("\nZONES")
print(df["zone"].unique())

print("\nCORRIDORS")
print(df["corridor"].dropna().unique()[:50])'''

'''import pickle

with open("models/feature_columns.pkl", "rb") as f:
    cols = pickle.load(f)

print(len(cols))
print(cols[:100])'''

import pickle

with open("models/event_cause_freq.pkl", "rb") as f:
    event_freq = pickle.load(f)

with open("models/corridor_freq.pkl", "rb") as f:
    corridor_freq = pickle.load(f)

print(event_freq)
print()
print(corridor_freq)