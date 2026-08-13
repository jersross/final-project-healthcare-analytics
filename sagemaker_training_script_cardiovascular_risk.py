import os
import json
import joblib
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# ==========================================================
# 1. AWS & SAGEMAKER INITIALIZATION
# ==========================================================
# PLACEHOLDER: Replace with your actual SageMaker Execution Role ARN if running outside SageMaker Studio
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

sagemaker_session = sagemaker.Session()
bucket = "mgmt-healthcare-analytics-jrms-final-project"
prefix = "patient-data/silver"

print(f"AWS Region: {sagemaker_session.boto_region_name}")
print(f"S3 Data Path: s3://{bucket}/{prefix}/")

# ==========================================================
# 2. LOAD DATA FROM S3 SILVER PARQUET
# ==========================================================
s3_path = f"s3://{bucket}/{prefix}/"
df = pd.read_parquet(s3_path)
print(f"Loaded {len(df)} patient records from Silver Layer.")

# Define Feature Sets
cat_cols = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
num_cols = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)']

# Encode Categorical Features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode Stage 1 Target
target_le = LabelEncoder()
df['CVD_Risk_Level_Encoded'] = target_le.fit_transform(df['CVD Risk Level'])

feature_cols = num_cols + cat_cols
X = df[feature_cols]
y_class = df['CVD_Risk_Level_Encoded']
y_reg = df['CVD Risk Score']

# Train / Test Split
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ==========================================================
# 3. TRAIN TWO-STAGE MODEL PIPELINE
# ==========================================================
print("Training Stage 1: HistGradientBoostingClassifier (CVD Risk Level)...")
clf = HistGradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_class_train)

# Stage 2 Regressor uses predicted risk level signal from Stage 1
X_train_reg = X_train.copy()
X_train_reg['CVD_Risk_Level_Signal'] = y_class_train

X_test_reg = X_test.copy()
X_test_reg['CVD_Risk_Level_Signal'] = clf.predict(X_test)

print("Training Stage 2: HistGradientBoostingRegressor (CVD Risk Score)...")
reg = HistGradientBoostingRegressor(random_state=42)
reg.fit(X_train_reg, y_reg_train)

# Metrics Evaluation
reg_preds = reg.predict(X_test_reg)
print(f"Model Test R2 Score: {r2_score(y_reg_test, reg_preds):.4f}")
print(f"Model Test RMSE: {np.sqrt(mean_squared_error(y_reg_test, reg_preds)):.4f}")

# ==========================================================
# 4. SAVE MODEL ARTIFACTS & INFERENCE CODE
# ==========================================================
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(clf, "model_artifacts/stage1_clf.joblib")
joblib.dump(reg, "model_artifacts/stage2_reg.joblib")
joblib.dump(label_encoders, "model_artifacts/label_encoders.joblib")
joblib.dump(target_le, "model_artifacts/target_le.joblib")

# Create code/inference.py for SageMaker Endpoint Handler
os.makedirs("model_artifacts/code", exist_ok=True)
with open("model_artifacts/code/inference.py", "w") as f:
    f.write('''
import os
import json
import joblib
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "stage1_clf.joblib"))
    reg = joblib.load(os.path.join(model_dir, "stage2_reg.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    target_le = joblib.load(os.path.join(model_dir, "target_le.joblib"))
    return {"clf": clf, "reg": reg, "encoders": label_encoders, "target_le": target_le}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_df, model_dict):
    clf = model_dict["clf"]
    reg = model_dict["reg"]
    encoders = model_dict["encoders"]
    target_le = model_dict["target_le"]

    # Transform categorical inputs
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # Stage 1 Prediction
    feature_cols = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 'Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
    input_features = input_df[feature_cols]
    
    pred_class_enc = clf.predict(input_features)[0]
    pred_class_label = target_le.inverse_transform([pred_class_enc])[0]

    # Stage 2 Prediction
    input_features_reg = input_features.copy()
    input_features_reg['CVD_Risk_Level_Signal'] = pred_class_enc
    pred_score = reg.predict(input_features_reg)[0]

    return {
        "predicted_risk_level": str(pred_class_label),
        "predicted_risk_score": round(float(pred_score), 2)
    }

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
''')

# Tar and upload model to S3
import tarfile
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model_artifacts/stage1_clf.joblib", arcname="stage1_clf.joblib")
    tar.add("model_artifacts/stage2_reg.joblib", arcname="stage2_reg.joblib")
    tar.add("model_artifacts/label_encoders.joblib", arcname="label_encoders.joblib")
    tar.add("model_artifacts/target_le.joblib", arcname="target_le.joblib")
    tar.add("model_artifacts/code/inference.py", arcname="code/inference.py")

model_s3_uri = sagemaker_session.upload_data("model.tar.gz", bucket=bucket, key_prefix="sagemaker/models")
print(f"Uploaded model tarball to: {model_s3_uri}")

# ==========================================================
# 5. DEPLOY REAL-TIME SAGEMAKER ENDPOINT
# ==========================================================
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data=model_s3_uri,
    role=ROLE_ARN,
    entry_point="code/inference.py",
    framework_version="1.2-1",
    py_version="py3"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium", # Cost-optimized instance
    endpoint_name="cvd-risk-prediction-endpoint"
)

print(f"Endpoint successfully deployed: {predictor.endpoint_name}")