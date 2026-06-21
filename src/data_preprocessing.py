import pandas as pd

df = pd.read_csv("data/Astram_event_data.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nEvent Type:")
print(df['event_type'].value_counts(dropna=False))

print("\nEvent Cause:")
print(df['event_cause'].value_counts(dropna=False))

print("\nPriority:")
print(df['priority'].value_counts(dropna=False))

print("\nRoad Closure:")
print(df['requires_road_closure'].value_counts(dropna=False))
# Datetime conversion

df['start_datetime'] = pd.to_datetime(
    df['start_datetime'],
    errors='coerce'
)

df['closed_datetime'] = pd.to_datetime(
    df['closed_datetime'],
    errors='coerce'
)

# Duration in minutes

df['duration_minutes'] = (
    df['closed_datetime'] -
    df['start_datetime']
).dt.total_seconds() / 60


print("\nDuration Statistics:")
print(df['duration_minutes'].describe())

print("\nRows with duration:")
print(df['duration_minutes'].notna().sum())