<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Hotel Booking Cancellation: Predictive Revenue Management Pipeline
Local Production Version with Full Evaluation, Artifact Export & Visual Dashboards
=======
"""
Hotel Booking Cancellation: Predictive Revenue Management Pipeline
>>>>>>> 9e69924
"""

import pandas as pd
import numpy as np
import optuna
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import warnings

warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 1. LOAD DATASET
print("Step 1: Loading 'Hotel Booking Cancellations.csv'...")
df = pd.read_csv('Hotel Booking Cancellations.csv')

# Standardize column headers: remove spaces, convert to lowercase
df.columns = [re.sub(r'\s+', '_', col.strip()).lower() for col in df.columns]

# Create binary target column (1 = Canceled, 0 = Not Canceled)
if 'booking_status' in df.columns:
    df['target'] = df['booking_status'].apply(lambda x: 1 if str(x).lower() == 'canceled' else 0)

# 2. DATA PREPARATION & FEATURE ENGINEERING
def prepare_ml_data(data):
    df_ml = data.copy()

    # Temporal feature engineering
    if 'date_of_reservation' in df_ml.columns:
        df_ml['date_dt'] = pd.to_datetime(df_ml['date_of_reservation'], errors='coerce')
        mode_date = df_ml['date_dt'].mode()[0] if not df_ml['date_dt'].mode().empty else pd.Timestamp('2025-01-01')
        df_ml['date_dt'] = df_ml['date_dt'].fillna(mode_date)
        df_ml['res_month'] = df_ml['date_dt'].dt.month
        df_ml['res_dayofweek'] = df_ml['date_dt'].dt.dayofweek
        df_ml.drop(['date_of_reservation', 'date_dt'], axis=1, inplace=True, errors='ignore')

    for col in ['booking_id', 'booking_status']:
        if col in df_ml.columns:
            df_ml.drop(col, axis=1, inplace=True)

    # Cast numeric columns to prevent string concatenation issues
    num_cols = [
        'number_of_adults', 'number_of_children', 'number_of_weekend_nights',
        'number_of_week_nights', 'car_parking_space', 'lead_time',
        'repeated', 'p-c', 'p-not-c', 'average_price', 'special_requests'
    ]
    for c in num_cols:
        if c in df_ml.columns:
            df_ml[c] = pd.to_numeric(df_ml[c], errors='coerce').fillna(0)

    # Engineered interaction features
    df_ml['total_guests'] = df_ml['number_of_adults'] + df_ml['number_of_children']
    df_ml['total_stay'] = df_ml['number_of_weekend_nights'] + df_ml['number_of_week_nights']
    df_ml['price_per_person'] = df_ml['average_price'] / (df_ml['total_guests'] + 0.1)

    return df_ml

print("Step 2: Cleaning data and building ML features...")
df_processed = prepare_ml_data(df)


# 3. SPLIT, ENCODE & SCALE
X = df_processed.drop('target', axis=1)
y = df_processed['target']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Native Scikit-Learn Target Encoder
te = TargetEncoder(smooth="auto", cv=5, random_state=42)
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

if cat_cols:
    X_train_encoded[cat_cols] = te.fit_transform(X_train[cat_cols], y_train)
    X_test_encoded[cat_cols] = te.transform(X_test[cat_cols])

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_encoded)
X_test_sc = scaler.transform(X_test_encoded)

print("Step 3: Balancing dataset with SMOTE...")
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_sc, y_train)


# 4. OPTUNA HYPERPARAMETER OPTIMIZATION
def objective(trial):
    n = trial.suggest_int('n_estimators', 100, 200)
    d = trial.suggest_int('max_depth', 10, 20)
    model = ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=42, n_jobs=-1)
    return cross_val_score(model, X_res, y_res, n_jobs=-1, cv=3, scoring='roc_auc').mean()

print("Step 4: Running Optuna parameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)


# 5. STACKING ENSEMBLE TRAINING
print("Step 5: Training final Stacking Ensemble...")
stack_clf = StackingClassifier(
    estimators=[
        ('et', ExtraTreesClassifier(**study.best_params, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
stack_clf.fit(X_res, y_res)

# 6. COMPREHENSIVE MODEL EVALUATION
y_pred = stack_clf.predict(X_test_sc)
y_proba = stack_clf.predict_proba(X_test_sc)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("           DETAILED PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy Score  : {acc * 100:.2f}% ({acc:.4f})")
print(f"Precision Score : {prec * 100:.2f}% ({prec:.4f})")
print(f"Recall Score    : {rec * 100:.2f}% ({rec:.4f})")
print(f"F1-Score        : {f1 * 100:.2f}% ({f1:.4f})")
print(f"ROC-AUC Score   : {roc:.4f}")
print("-" * 50)
print("Confusion Matrix:\n", cm)
print("-" * 50)
print("Full Classification Report:\n", classification_report(y_test, y_pred))
print("="*50)


# 7. EXPORT MODEL ARTIFACTS

joblib.dump(stack_clf, 'hotel_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(te, 'encoder.pkl')
print("\nArtifacts successfully exported: 'hotel_model.pkl', 'scaler.pkl', and 'encoder.pkl'")


# 8. VISUAL DASHBOARDS GENERATION

print("\nStep 8: Generating Visual Dashboards...")

# --- DASHBOARD 1: MODEL EVALUATION & PERFORMANCE ---
# ==========================================================
# 8. VISUAL DASHBOARDS GENERATION
# ==========================================================
print("\nStep 8: Generating Visual Dashboards...")

# Ensure plotting dataframe columns are strictly numeric
df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce').fillna(0)
df['special_requests'] = pd.to_numeric(df['special_requests'], errors='coerce').fillna(0)
df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0)

# --- DASHBOARD 1: MODEL EVALUATION & PERFORMANCE ---
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('MODEL PERFORMANCE & EVALUATION MATRIX', fontsize=18, fontweight='bold', y=0.98)

# 1.1 Metrics Summary Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [acc, prec, rec, f1, roc]
colors = ['#2b5c8f', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

bars = axes1[0, 0].bar(metrics_names, metrics_values, color=colors, width=0.55, edgecolor='black')
axes1[0, 0].set_ylim(0, 1.15)
axes1[0, 0].set_title('Overall Performance Scores', fontsize=14, fontweight='bold')
axes1[0, 0].set_ylabel('Score (0.0 to 1.0)')
for bar in bars:
    yval = bar.get_height()
    axes1[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 1.2 Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes1[0, 1],
            xticklabels=['Confirmed', 'Canceled'], yticklabels=['Confirmed', 'Canceled'], annot_kws={'size': 14, 'weight': 'bold'})
axes1[0, 1].set_title('Confusion Matrix (Ground Truth vs Prediction)', fontsize=14, fontweight='bold')
axes1[0, 1].set_xlabel('Predicted Booking Status', fontweight='bold')
axes1[0, 1].set_ylabel('Actual Booking Status', fontweight='bold')

# 1.3 ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
axes1[1, 0].plot(fpr, tpr, color='#1f77b4', lw=3, label=f'Stacking Classifier (AUC = {roc_auc_val:.3f})')
axes1[1, 0].plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Guess')
axes1[1, 0].fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
axes1[1, 0].set_xlim([0.0, 1.0])
axes1[1, 0].set_ylim([0.0, 1.05])
axes1[1, 0].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
axes1[1, 0].set_ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
axes1[1, 0].set_title('ROC-AUC Curve', fontsize=14, fontweight='bold')
axes1[1, 0].legend(loc="lower right")

# 1.4 Cancellation Probability Distribution by Class
axes1[1, 1].hist(y_proba[y_test == 0], bins=25, alpha=0.6, label='Actual: Not Canceled', color='seagreen', edgecolor='black')
axes1[1, 1].hist(y_proba[y_test == 1], bins=25, alpha=0.6, label='Actual: Canceled', color='crimson', edgecolor='black')
axes1[1, 1].axvline(0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
axes1[1, 1].set_title('Predicted Risk Distribution by True Class', fontsize=14, fontweight='bold')
axes1[1, 1].set_xlabel('Predicted Probability of Cancellation', fontweight='bold')
axes1[1, 1].set_ylabel('Number of Bookings', fontweight='bold')
axes1[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_evaluation_matrix.png', dpi=300)
print("Saved: 'model_evaluation_matrix.png'")

# --- DASHBOARD 2: DOMAIN INSIGHTS & RELATIONSHIPS ---
fig2, axes2 = plt.subplots(3, 2, figsize=(18, 20))
fig2.suptitle('BEHAVIORAL & OPERATIONAL CANCELLATION DRIVERS', fontsize=18, fontweight='bold', y=0.99)

# 2.1 Booking Channel (Market Segment) Reliability Index
channel_risk = df.groupby('market_segment_type')['target'].mean().sort_values(ascending=False) * 100
sns.barplot(x=channel_risk.values, y=channel_risk.index, ax=axes2[0, 0], palette='Reds_r', edgecolor='black')
axes2[0, 0].set_title('1. Booking Channel Cancellation Risk (%)', fontsize=13, fontweight='bold')
axes2[0, 0].set_xlabel('Cancellation Rate (%)', fontweight='bold')
for i, v in enumerate(channel_risk.values):
    axes2[0, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

# 2.2 Lead Time vs Cancellation Probability
df['lead_bin'] = pd.cut(df['lead_time'], bins=[-1, 14, 30, 90, 180, 700], 
                        labels=['0-14 days', '15-30 days', '1-3 months', '3-6 months', '6+ months'])
lead_risk = df.groupby('lead_bin', observed=False)['target'].mean() * 100
sns.barplot(x=lead_risk.index, y=lead_risk.values, ax=axes2[0, 1], palette='Oranges_r', edgecolor='black')
axes2[0, 1].set_title('2. Lead Time vs. Cancellation Risk', fontsize=13, fontweight='bold')
axes2[0, 1].set_ylabel('Cancellation Rate (%)', fontweight='bold')
for i, v in enumerate(lead_risk.values):
    axes2[0, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

# 2.3 Special Requests / Customer Engagement Impact
req_risk = df.groupby('special_requests')['target'].mean() * 100
axes2[1, 0].plot(req_risk.index, req_risk.values, marker='o', color='#2ca02c', linewidth=3.5, markersize=10)
axes2[1, 0].fill_between(req_risk.index, req_risk.values, color='#2ca02c', alpha=0.15)
axes2[1, 0].set_title('3. Guest Engagement Effect (Special Requests vs Risk)', fontsize=13, fontweight='bold')
axes2[1, 0].set_xlabel('Number of Special Requests', fontweight='bold')
axes2[1, 0].set_ylabel('Cancellation Probability (%)', fontweight='bold')
for x, y in zip(req_risk.index, req_risk.values):
    axes2[1, 0].text(x, y + 1.2, f'{y:.1f}%', ha='center', fontweight='bold')

# 2.4 Meal Plan Specific Cancellation Chances
meal_risk = df.groupby('type_of_meal')['target'].mean().sort_values(ascending=False) * 100
sns.barplot(x=meal_risk.index, y=meal_risk.values, ax=axes2[1, 1], palette='Purples_r', edgecolor='black')
axes2[1, 1].set_title('4. Meal Plan Selection vs. Cancellation Chance', fontsize=13, fontweight='bold')
axes2[1, 1].set_ylabel('Cancellation Rate (%)', fontweight='bold')
for i, v in enumerate(meal_risk.values):
    axes2[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

# 2.5 Room Type Cancellation Rate
room_risk = df.groupby('room_type')['target'].mean().sort_values(ascending=False) * 100
sns.barplot(x=room_risk.index, y=room_risk.values, ax=axes2[2, 0], palette='Blues_r', edgecolor='black')
axes2[2, 0].set_title('5. Room Category vs. Cancellation Chance', fontsize=13, fontweight='bold')
axes2[2, 0].set_ylabel('Cancellation Rate (%)', fontweight='bold')
for i, v in enumerate(room_risk.values):
    axes2[2, 0].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

# 2.6 Overall Volume vs Cancellation Distribution
if 'booking_status' in df.columns:
    req_counts = df.groupby(['special_requests', 'booking_status']).size().unstack(fill_value=0)
    req_counts.plot(kind='bar', stacked=True, ax=axes2[2, 1], color=['#d62728', '#2ca02c'], edgecolor='black')
    axes2[2, 1].set_title('6. Total Booking Volume by Engagement & Status', fontsize=13, fontweight='bold')
    axes2[2, 1].set_xlabel('Number of Special Requests', fontweight='bold')
    axes2[2, 1].set_ylabel('Reservation Count', fontweight='bold')
    axes2[2, 1].legend(['Canceled', 'Confirmed'], title='Outcome')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('behavioral_insights.png', dpi=300)
print("Saved: 'behavioral_insights.png'")

# --- DASHBOARD 3: REVENUE RECOVERY & FEASIBLE SOLUTIONS ---
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle('FINANCIAL RECOVERY & INTERVENTION SIMULATION', fontsize=18, fontweight='bold', y=1.02)

# 3.1 Financial Impact ($ Lost vs Recovered)
total_loss = X_test['average_price'][y_test == 1].sum()
caught_cancellations_val = X_test['average_price'][(y_test == 1) & (y_pred == 1)].sum()
recovered_amount = caught_cancellations_val * 0.60
unavoidable_loss = total_loss - recovered_amount

categories = ['Total At-Risk Revenue', 'Recoverable via AI System', 'Unavoidable Net Loss']
values = [total_loss, recovered_amount, unavoidable_loss]
bar_colors = ['#d9534f', '#5cb85c', '#f0ad4e']

b3 = axes3[0].bar(categories, values, color=bar_colors, edgecolor='black', width=0.5)
axes3[0].set_title('3.1 Revenue Recovery Impact ($)', fontsize=14, fontweight='bold')
axes3[0].set_ylabel('Revenue Amount in USD ($)', fontweight='bold')
for bar in b3:
    yval = bar.get_height()
    axes3[0].text(bar.get_x() + bar.get_width()/2.0, yval + (total_loss * 0.02), f'${int(yval):,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 3.2 Feasible Solutions: Confirmation Lift from Targeted Interventions
interventions = [
    'Baseline (No Action)',
    'Reminder Email (+5%)',
    'Free Breakfast Offer (+15%)',
    '10% Discount on Add-ons (+25%)',
    'Non-refundable 15% Deposit (+40%)'
]
baseline_confirmation = 100 - (y_proba[y_proba > 0.60].mean() * 100) if (y_proba > 0.60).any() else 40.0
confirmation_lifts = [
    baseline_confirmation,
    baseline_confirmation + 5.0,
    baseline_confirmation + 15.0,
    baseline_confirmation + 25.0,
    baseline_confirmation + 40.0
]

b4 = axes3[1].barh(interventions, confirmation_lifts, color='#337ab7', edgecolor='black', height=0.55)
axes3[1].set_title('3.2 Feasible Action Lift: Confirmation % on High-Risk Bookings', fontsize=14, fontweight='bold')
axes3[1].set_xlabel('Projected Confirmation Rate (%)', fontweight='bold')
axes3[1].set_xlim(0, 100)
for bar in b4:
    xval = bar.get_width()
    axes3[1].text(xval + 1.5, bar.get_y() + bar.get_height()/2.0, f'{xval:.1f}%', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('revenue_and_solutions.png', dpi=300)
print("Saved: 'revenue_and_solutions.png'")

print("\nProcessing complete. All 3 dashboard images saved successfully.")
<<<<<<< HEAD
plt.show()
=======
plt.show()
>>>>>>> 9e69924
