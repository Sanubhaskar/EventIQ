EventIQ – AI-Powered Traffic Risk Prediction System

Team SKAM

* B. Sahana
* N. Krithikha
* Aswini Venkatesan
* K. K. Madhumitha

---

Overview

EventIQ is an AI-powered traffic risk prediction and decision-support system designed to help traffic management authorities proactively identify incidents that are likely to result in prolonged disruptions.

Using historical Bengaluru traffic event data, EventIQ analyzes temporal, spatial, and operational factors to classify traffic events as either **High Risk** or **Low Risk**. The system also provides interactive tools for scenario analysis and traffic intelligence.

---

Problem Statement

Traffic authorities often respond after congestion has already escalated. Incidents such as vehicle breakdowns, road construction, accidents, and water logging can significantly impact traffic flow and resource allocation.

EventIQ aims to support proactive traffic management by predicting high-risk incidents before they develop into severe disruptions.

---

Features

Risk Predictor

Predicts whether an incoming traffic event is likely to become a prolonged disruption.

What-If Simulator

Allows operators to compare different intervention scenarios and evaluate how operational decisions affect disruption risk.

Traffic Intelligence Dashboard

Provides model performance metrics, event distributions, feature importance insights, and operational intelligence.

---

Dataset

Source: Bengaluru Traffic Event Dataset

Original Dataset

* 8,173 traffic event records

Training Dataset

* 2,460 validated records

Data Processing

* Timestamp conversion
* Duration calculation
* Removal of invalid records
* Outlier handling
* Feature engineering

---

Feature Engineering

Temporal Features

* Hour of Day
* Day of Week
* Month
* Weekend Indicator
* Peak Hour Indicator

Operational Features

* Event Type
* Event Cause
* Priority
* Road Closure Requirement

Spatial Features

* Corridor
* Zone
* Non-Corridor Indicator

Frequency-Based Features

* Event Cause Frequency
* Corridor Frequency

---

Machine Learning Pipeline

Initial Approach

Multi-class Classification

* Low
* Medium
* High
* Critical

Result:

* Accuracy ≈ 42%

Final Approach

Binary Classification

* High Risk
* Low Risk

Model

XGBoost Classifier

Reasons for Selection:

* Strong performance on tabular datasets
* Handles class imbalance effectively
* Captures non-linear relationships
* Supports explainability through feature importance

Model Performance

| Metric                  | Value         |
| ----------------------- | ------------- |
| Accuracy                | 88%           |
| ROC-AUC                 | 0.944         |
| High-Risk Recall        | 89%           |
| Cross-Validated ROC-AUC | 0.941 ± 0.007 |

Technology Stack

Programming Language

* Python

Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* SHAP

Tools

* Git
* GitHub


## Project Structure

EventIQ/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── feature_engineering.py
│   ├── train_binary_model.py
│   └── error_analysis.py
│
├── models/
│   ├── eventiq_binary_model.pkl
│   ├── feature_columns.pkl
│   ├── label_encoder.pkl
│   ├── event_cause_freq.pkl
│   └── corridor_freq.pkl
│
└── data/
    └── cleaned_events.csv
    
Installation: pip install -r requirements.txt

To Run the Application: streamlit run app.py

## Future Enhancements

* Real-time traffic feed integration
* Weather data integration
* Planned-event calendar integration
* Advanced scenario simulation
* Dynamic route recommendations

Conclusion

EventIQ demonstrates how machine learning can support proactive traffic operations by identifying high-risk incidents early, improving resource allocation, and enabling data-driven decision-making for urban traffic management systems.
