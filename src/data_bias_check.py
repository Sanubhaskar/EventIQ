import pandas as pd

df = pd.read_csv("data/Astram_event_data.csv")

with_duration = df[df["closed_datetime"].notna()]
without_duration = df[df["closed_datetime"].isna()]

print("WITH duration:", len(with_duration))
print("WITHOUT duration:", len(without_duration))

print("\n=== EVENT CAUSE ===")

print("\nWITH duration")
print(
    with_duration["event_cause"]
    .value_counts(normalize=True)
    .round(3)
)

print("\nWITHOUT duration")
print(
    without_duration["event_cause"]
    .value_counts(normalize=True)
    .round(3)
)

print("\n=== PRIORITY ===")

print("\nWITH duration")
print(
    with_duration["priority"]
    .value_counts(normalize=True)
    .round(3)
)

print("\nWITHOUT duration")
print(
    without_duration["priority"]
    .value_counts(normalize=True)
    .round(3)
)

print("\n=== EVENT TYPE ===")

print("\nWITH duration")
print(
    with_duration["event_type"]
    .value_counts(normalize=True)
    .round(3)
)

print("\nWITHOUT duration")
print(
    without_duration["event_type"]
    .value_counts(normalize=True)
    .round(3)
)