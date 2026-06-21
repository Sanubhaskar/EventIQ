import pandas as pd

# Load data
df = pd.read_csv("data/Astram_event_data.csv")

# Datetime conversion
df['start_datetime'] = pd.to_datetime(
    df['start_datetime'],
    errors='coerce'
)

df['closed_datetime'] = pd.to_datetime(
    df['closed_datetime'],
    errors='coerce'
)

# Duration
df['duration_minutes'] = (
    df['closed_datetime']
    - df['start_datetime']
).dt.total_seconds() / 60

# Keep only rows with duration
df = df[df['duration_minutes'].notna()]

# Remove negative durations
df = df[df['duration_minutes'] >= 0]

# Remove extreme outliers
df = df[df['duration_minutes'] <= 1440]

# Create time features
df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek

# Create impact class
def impact_class(duration):

    if duration < 30:
        return "Low"

    elif duration < 90:
        return "Medium"

    elif duration < 180:
        return "High"

    return "Critical"

df['impact_class'] = df['duration_minutes'].apply(
    impact_class
)

df['month'] = df['start_datetime'].dt.month

df['is_weekend'] = (
    df['start_datetime'].dt.dayofweek >= 5
).astype(int)

df["is_peak_hour"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 10))
    |
    ((df["hour"] >= 17) & (df["hour"] <= 20))
).astype(int)

df["is_non_corridor"] = (
    df["corridor"] == "Non-corridor"
).astype(int)

freq = df["event_cause"].value_counts()

df["event_cause_freq"] = (
    df["event_cause"]
    .map(freq)
)

corridor_freq = df["corridor"].value_counts()

df["corridor_freq"] = (
    df["corridor"]
    .map(corridor_freq)
) 

print("Final rows:", len(df))

print("\nImpact Distribution:")
print(df['impact_class'].value_counts())

print("\nDuration Stats:")
print(df['duration_minutes'].describe())

# Save cleaned dataset
df.to_csv(
    "data/cleaned_events.csv",
    index=False
)
