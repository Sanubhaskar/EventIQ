import pandas as pd

df = pd.read_csv("data/cleaned_events.csv")

preds = pd.read_csv("data/error_analysis.csv")

df = df.reset_index()
df.rename(columns={"index": "original_index"}, inplace=True)

merged = pd.merge(
    preds,
    df,
    on="original_index"
)

false_negatives = merged[
    (merged["actual"] == 1)
    &
    (merged["predicted"] == 0)
]

false_positives = merged[
    (merged["actual"] == 0)
    &
    (merged["predicted"] == 1)
]

print("\nFALSE NEGATIVES:", len(false_negatives))

print("\nTop Causes:")
print(
    false_negatives["event_cause"]
    .value_counts()
)

print("\nTop Corridors:")
print(
    false_negatives["corridor"]
    .value_counts()
)

print("\n\nFALSE POSITIVES:", len(false_positives))

print("\nTop Causes:")
print(
    false_positives["event_cause"]
    .value_counts()
)

print("\nTop Corridors:")
print(
    false_positives["corridor"]
    .value_counts()
)