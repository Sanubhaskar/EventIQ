import pandas as pd
import pickle

df = pd.read_csv("data/cleaned_events.csv")

event_cause_freq = (
    df["event_cause"]
    .value_counts()
    .to_dict()
)

corridor_freq = (
    df["corridor"]
    .value_counts()
    .to_dict()
)

with open("models/event_cause_freq.pkl", "wb") as f:
    pickle.dump(event_cause_freq, f)

with open("models/corridor_freq.pkl", "wb") as f:
    pickle.dump(corridor_freq, f)

print("Mappings saved")