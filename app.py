import streamlit as st
import pandas as pd
import numpy as np
import pickle
with open("models/eventiq_binary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/event_cause_freq.pkl", "rb") as f:
    event_freq = pickle.load(f)

with open("models/corridor_freq.pkl", "rb") as f:
    corridor_freq = pickle.load(f)

page = st.sidebar.radio(
    "Navigation",
    [
        "Risk Predictor",
        "What-If Simulator",
        "Intelligence Dashboard"
    ]
)

def make_prediction(
    event_type,
    event_cause,
    priority,
    corridor,
    zone,
    road_closure,
    hour
):

    row = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    row["requires_road_closure"] = road_closure
    row["hour"] = hour

    day_of_week = 1
    month = 6

    row["day_of_week"] = day_of_week
    row["month"] = month

    row["is_weekend"] = 0

    row["is_peak_hour"] = int(
        (7 <= hour <= 10)
        or
        (17 <= hour <= 20)
    )

    row["event_cause_freq"] = (
        event_freq.get(event_cause, 0)
    )

    row["corridor_freq"] = (
        corridor_freq.get(corridor, 0)
    )

    row["is_non_corridor"] = int(
        corridor == "Non-corridor"
    )

    # Event Type

    if event_type == "unplanned":
        row["event_type_unplanned"] = 1

    # Event Cause

    cause_col = f"event_cause_{event_cause}"

    if cause_col in row.columns:
        row[cause_col] = 1

    # Priority

    if priority == "Low":
        row["priority_Low"] = 1

    # Corridor

    corridor_col = f"corridor_{corridor}"

    if corridor_col in row.columns:
        row[corridor_col] = 1

    # Zone

    zone_col = f"zone_{zone}"

    if zone_col in row.columns:
        row[zone_col] = 1

    prob = model.predict_proba(row)[0][1]

    if prob >= 0.7:
        prediction = "HIGH RISK"

    elif prob >= 0.3:
        prediction = "MODERATE RISK"

    else:
        prediction = "LOW RISK"

    return prediction, prob

if page == "Risk Predictor":

    st.title("🚦 EventIQ")
    st.subheader("Traffic Risk Prediction")

    event_type = st.selectbox(
        "Event Type",
        ["planned", "unplanned"]
    )

    event_cause = st.selectbox(
        "Event Cause",
        list(event_freq.keys())
    )

    priority = st.selectbox(
        "Priority",
        ["High", "Low"]
    )

    corridor = st.selectbox(
        "Corridor",
        list(corridor_freq.keys())
    )

    zone = st.selectbox(
        "Zone",
        [
            "Central Zone 1",
            "Central Zone 2",
            "East Zone 1",
            "East Zone 2",
            "North Zone 1",
            "North Zone 2",
            "South Zone 1",
            "South Zone 2",
            "West Zone 1",
            "West Zone 2"
        ]
    )

    road_closure = st.selectbox(
        "Road Closure",
        [0, 1]
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        9
    )

    if st.button("Predict"):

        pred, prob = make_prediction(
            event_type,
            event_cause,
            priority,
            corridor,
            zone,
            road_closure,
            hour
        )

        st.metric(
            "Risk Level",
            pred
        )

        st.metric(
            "Probability",
            f"{prob*100:.1f}%"
        )

        if prob >= 0.7:
            st.error("High likelihood of prolonged disruption")
        elif prob >= 0.3:
            st.warning("Moderate disruption risk")
        else:
            st.success("Low disruption risk")
if page == "What-If Simulator":

    st.title("🔄 What-If Simulator")

    st.write(
        "Evaluate how operational changes affect disruption risk."
    )

    st.subheader("Current Scenario")

    current_event_type = st.selectbox(
        "Current Event Type",
        ["planned", "unplanned"]
    )

    current_event_cause = st.selectbox(
        "Current Event Cause",
        list(event_freq.keys())
    )

    current_priority = st.selectbox(
        "Current Priority",
        ["High", "Low"]
    )

    current_corridor = st.selectbox(
        "Current Corridor",
        list(corridor_freq.keys())
    )

    current_zone = st.selectbox(
        "Current Zone",
        [
            "Central Zone 1",
            "Central Zone 2",
            "East Zone 1",
            "East Zone 2",
            "North Zone 1",
            "North Zone 2",
            "South Zone 1",
            "South Zone 2",
            "West Zone 1",
            "West Zone 2"
        ]
    )

    current_road_closure = st.selectbox(
        "Current Road Closure",
        [0, 1]
    )

    current_hour = st.slider(
        "Current Hour",
        0,
        23,
        9
    )

    st.divider()

    st.subheader("Modified Scenario")

    new_priority = st.selectbox(
        "Modified Priority",
        ["High", "Low"]
    )

    new_road_closure = st.selectbox(
        "Modified Road Closure",
        [0, 1]
    )

    new_hour = st.slider(
        "Modified Hour",
        0,
        23,
        9
    )

    new_event_cause = st.selectbox(
        "Modified Event Cause",
        list(event_freq.keys())
    )

    new_corridor = st.selectbox(
        "Modified Corridor",
        list(corridor_freq.keys())
    )

    if st.button("Run Simulation"):

        _, current_prob = make_prediction(
            current_event_type,
            current_event_cause,
            current_priority,
            current_corridor,
            current_zone,
            current_road_closure,
            current_hour
        )

        _, new_prob = make_prediction(
            current_event_type,
            new_event_cause,
            new_priority,
            new_corridor,
            current_zone,
            new_road_closure,
            new_hour
        )
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Current Risk",
                f"{current_prob * 100:.1f}%"
            )

        with col2:
            st.metric(
                "New Risk",
                f"{new_prob * 100:.1f}%"
            )

        change = (
            current_prob - new_prob
        ) * 100

        st.metric(
            "Risk Reduction",
            f"{change:.1f}%"
        )

        if change > 10:

            st.success(
                "This intervention significantly reduces disruption risk."
            )

        elif change > 0:

            st.info(
                "This intervention provides moderate improvement."
            )

        elif change == 0:

            st.warning(
                "No change detected."
            )

        else:

            st.error(
                "This modification increases disruption risk."
            )

        st.subheader("Decision Support")

        if new_prob < current_prob:

            st.write(
                """
                Recommended Action:

                The modified scenario produces a lower risk score.
                Consider implementing these operational changes
                to reduce the likelihood of a prolonged disruption.
                """
            )

        else:

            st.write(
                """
                Recommended Action:

                The modified scenario does not improve the risk score.
                Consider alternative interventions.
                """
            )


if page == "Intelligence Dashboard":

    st.title("📊 Traffic Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ROC-AUC",
        "0.944"
    )

    col2.metric(
        "CV AUC",
        "0.941 ± 0.007"
    )

    col3.metric(
        "High-Risk Recall",
        "89%"
    )

    st.subheader("Event Cause Distribution")

    st.bar_chart(
        pd.Series(event_freq)
    )

    st.subheader("Top Insights")

    st.info(
        """
        • Vehicle breakdowns are the strongest predictor.

        • Peak-hour incidents show elevated risk.

        • Certain corridors repeatedly exhibit prolonged disruptions.

        • EventIQ achieved ROC-AUC 0.944.
        """
    )