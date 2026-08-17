# Title: CVD Risk Prediction Pipeline & Artifact Packaging (Console-Ready)

# Set environment variables for sagemaker_studio imports
import os
os.environ['DataZoneProjectId'] = 'b25eihxw14kuqv'
os.environ['DataZoneDomainId'] = 'dzd-4s898z6ckmrexz'
os.environ['DataZoneEnvironmentId'] = '6ol7a5mhhwtg2v'
os.environ['DataZoneDomainRegion'] = 'us-east-1'

# Create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "b25eihxw14kuqv",
                "DataZoneDomainId": "dzd-4s898z6ckmrexz",
                "DataZoneEnvironmentId": "6ol7a5mhhwtg2v",
                "DataZoneDomainRegion": "us-east-1",
            }
        }
    return _resource_metadata

metadata = _get_resource_metadata()

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================
from typing import Optional

def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")

_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# ==========================================================
# IMPORTS FOR ML MODELING AND SAGEMAKER PACKAGING
# ==========================================================
import re
import time
import json
import joblib
import tarfile
import pandas as pd
import numpy as np
import boto3
import sagemaker
import sklearn
from sagemaker.sklearn.model import SKLearnModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# ==========================================================
# 0. SCIKIT-LEARN VERSION GUARD
# ==========================================================
# The SKLearnModel below is pinned to framework_version="1.2-1", meaning the
# SageMaker inference container has scikit-learn 1.2.1 installed. joblib/pickle
# serialize a reference to the exact internal module path each object lives in
# at dump time (e.g. sklearn._loss) -- that path has moved across sklearn
# releases. If this notebook's kernel has a different scikit-learn version,
# the model trains and dumps fine here, but the container fails to unpickle it
# at deploy time (ModuleNotFoundError: No module named '_loss'), and the
# endpoint never leaves "Creating". Catch that here instead of at deploy time.
REQUIRED_SKLEARN_VERSION = "1.2.1"
if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
    raise RuntimeError(
        f"scikit-learn {sklearn.__version__} is installed, but the SageMaker "
        f"inference container is pinned to {REQUIRED_SKLEARN_VERSION} "
        f"(framework_version=\"1.2-1\" below). Run:\n"
        f"    pip install scikit-learn=={REQUIRED_SKLEARN_VERSION}\n"
        f"then RESTART THE KERNEL and re-run this script from the top before "
        f"training -- otherwise the model artifacts will fail to load in the "
        f"container and the endpoint will never reach InService."
    )
print(f"scikit-learn version check passed: {sklearn.__version__}")

# ==========================================================
# 1. AWS & SAGEMAKER INITIALIZATION
# ==========================================================
sagemaker_session = sagemaker.Session()
bucket = "mgmt-healthcare-analytics-jrms-final-project"
prefix = "patient-data/silver"

try:
    ROLE_ARN = sagemaker.get_execution_role()
except Exception:
    ROLE_ARN = "arn:aws:iam::060276830641:role/SageMaker-Healthcare-Analytics-Role"

print(f"AWS Region: {sagemaker_session.boto_region_name}")
print(f"Using IAM Role: {ROLE_ARN}")
print(f"S3 Data Path: s3://{bucket}/{prefix}/")

# ==========================================================
# 2. LOAD DATA FROM S3 SILVER PARQUET & FEATURE ENGINEERING
# ==========================================================
s3_path = f"s3://{bucket}/{prefix}/"
df = pd.read_parquet(s3_path)
print(f"Loaded {len(df)} patient records from Silver Layer.")

# Clean GlueParquet column suffixes (e.g., 'Systolic_BP#12' -> 'Systolic BP')
column_mapping = {}
for c in df.columns:
    clean_c = re.sub(r'#\d+$', '', c)
    c_lower = clean_c.lower().replace('_', ' ').strip()
    
    if 'sex' in c_lower: column_mapping[c] = 'Sex'
    elif 'weight' in c_lower and 'was' not in c_lower: column_mapping[c] = 'Weight_kg'
    elif 'height' in c_lower and 'm' in c_lower and 'was' not in c_lower and 'cm' not in c_lower: column_mapping[c] = 'Height_m'
    elif 'systolic' in c_lower and 'was' not in c_lower: column_mapping[c] = 'Systolic BP'
    elif 'diastolic' in c_lower and 'was' not in c_lower: column_mapping[c] = 'Diastolic BP'
    elif ('total' in c_lower or 'cholesterol' in c_lower) and 'was' not in c_lower: column_mapping[c] = 'Total Cholesterol (mg/dL)'
    elif 'hdl' in c_lower and 'was' not in c_lower: column_mapping[c] = 'HDL (mg/dL)'
    elif ('sugar' in c_lower or 'fasting' in c_lower) and 'was' not in c_lower: column_mapping[c] = 'Fasting Blood Sugar (mg/dL)'
    elif 'smok' in c_lower: column_mapping[c] = 'Smoking Status'
    elif 'diabe' in c_lower and 'was' not in c_lower: column_mapping[c] = 'Diabetes Status'
    elif 'activity' in c_lower or 'physical' in c_lower: column_mapping[c] = 'Physical Activity Level'
    elif 'family' in c_lower or 'history' in c_lower: column_mapping[c] = 'Family History of CVD'
    elif 'level' in c_lower and 'risk' in c_lower: column_mapping[c] = 'CVD Risk Level'
    elif 'score' in c_lower and 'risk' in c_lower and 'was' not in c_lower: column_mapping[c] = 'CVD Risk Score'

df = df.rename(columns=column_mapping)

# Derive missing Age feature from age_band if Age is absent
if 'Age' not in df.columns and 'age_band' in df.columns:
    df['Age'] = df['age_band'].str.extract(r'(\d+)').astype(float).fillna(50.0)
elif 'Age' not in df.columns:
    df['Age'] = 50.0

# Derive missing BMI feature from Weight_kg and Height_m
if 'BMI' not in df.columns and 'Weight_kg' in df.columns and 'Height_m' in df.columns:
    df['BMI'] = df['Weight_kg'] / (df['Height_m'] ** 2)
elif 'BMI' not in df.columns:
    df['BMI'] = 25.0

cat_cols = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
num_cols = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)']

missing_cols = [c for c in num_cols + cat_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"The following required columns were not found in S3 Parquet dataset: {missing_cols}")

# Encode Categorical Features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode Target
target_le = LabelEncoder()
df['CVD_Risk_Level_Encoded'] = target_le.fit_transform(df['CVD Risk Level'])

feature_cols = num_cols + cat_cols
X = df[feature_cols]
y_class = df['CVD_Risk_Level_Encoded']
y_reg = df['CVD Risk Score']

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ==========================================================
# 3. TRAIN TWO-STAGE MODEL PIPELINE
# ==========================================================
print("Training Stage 1: HistGradientBoostingClassifier (CVD Risk Level)...")
clf = HistGradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_class_train)

X_train_reg = X_train.copy()
X_train_reg['CVD_Risk_Level_Signal'] = y_class_train

X_test_reg = X_test.copy()
X_test_reg['CVD_Risk_Level_Signal'] = clf.predict(X_test)

print("Training Stage 2: HistGradientBoostingRegressor (CVD Risk Score)...")
reg = HistGradientBoostingRegressor(random_state=42)
reg.fit(X_train_reg, y_reg_train)

reg_preds = reg.predict(X_test_reg)
print(f"Model Test R2 Score: {r2_score(y_reg_test, reg_preds):.4f}")
print(f"Model Test RMSE: {np.sqrt(mean_squared_error(y_reg_test, reg_preds)):.4f}")

# ==========================================================
# 4. SAVE MODEL ARTIFACTS & INFERENCE CODE
# ==========================================================
os.makedirs("model_artifacts/code", exist_ok=True)
joblib.dump(clf, "model_artifacts/stage1_clf.joblib")
joblib.dump(reg, "model_artifacts/stage2_reg.joblib")
joblib.dump(label_encoders, "model_artifacts/label_encoders.joblib")
joblib.dump(target_le, "model_artifacts/target_le.joblib")

# Create model_artifacts/code/inference.py for SageMaker Endpoint Handler
with open("model_artifacts/code/inference.py", "w") as f:
    f.write('''import os
import json
import joblib
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info("Loading model artifacts from: %s", model_dir)
    try:
        clf = joblib.load(os.path.join(model_dir, "stage1_clf.joblib"))
        reg = joblib.load(os.path.join(model_dir, "stage2_reg.joblib"))
        label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
        target_le = joblib.load(os.path.join(model_dir, "target_le.joblib"))
        logger.info("Successfully loaded all model artifacts.")
        return {"clf": clf, "reg": reg, "encoders": label_encoders, "target_le": target_le}
    except Exception as e:
        logger.exception("Failed to load model artifacts")
        raise e

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame(data)
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

    # Define feature ordering
    feature_cols = [
        'Age', 'BMI', 'Systolic BP', 'Diastolic BP', 
        'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 
        'Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD'
    ]
    input_features = input_df[feature_cols]
    
    # Stage 1 Prediction
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

# Tar and upload model package to S3
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model_artifacts/stage1_clf.joblib", arcname="stage1_clf.joblib")
    tar.add("model_artifacts/stage2_reg.joblib", arcname="stage2_reg.joblib")
    tar.add("model_artifacts/label_encoders.joblib", arcname="label_encoders.joblib")
    tar.add("model_artifacts/target_le.joblib", arcname="target_le.joblib")
    tar.add("model_artifacts/code/inference.py", arcname="code/inference.py")

model_s3_uri = sagemaker_session.upload_data("model.tar.gz", bucket=bucket, key_prefix="sagemaker/models")

# ==========================================================
# 5. SAGEMAKER MODEL PACKAGING FOR MANUAL CONSOLE DEPLOYMENT (FREE TIER)
# ==========================================================
sklearn_model = SKLearnModel(
    model_data=model_s3_uri,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="model_artifacts/code",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Set to Free Tier eligible instance type
image_uri = sklearn_model.prepare_container_def(instance_type="ml.t2.medium")["Image"]

# Register the Model resource in SageMaker via the SDK. This is what your
# deploy notebook's describe_model("cvd-risk-model") call is looking for --
# the SDK also sets the container's SAGEMAKER_PROGRAM and
# SAGEMAKER_SUBMODULE_DIRECTORY environment variables correctly and
# automatically, avoiding a hand-typed mismatch when creating the Model
# resource manually in the console.
sklearn_model.create(instance_type="ml.t2.medium")
print(f"Model registered in SageMaker: {sklearn_model.name}")

print("\n" + "="*70)
print("SAGEMAKER CONSOLE MANUAL DEPLOYMENT SUMMARY (FREE TIER ELIGIBLE)")
print("="*70)
print(f"1. Model Artifact S3 Location:\n   {model_s3_uri}\n")
print(f"2. Docker Image URI (Framework Container):\n   {image_uri}\n")
print(f"3. IAM Execution Role ARN:\n   {ROLE_ARN}\n")
print(f"4. Registered SageMaker Model Name:\n   {sklearn_model.name}\n")
print(f"5. Target Instance Type (Free Tier):\n   ml.t2.medium\n")
print(f"6. Environment Variable Requirements (for reference, already set by SDK above):\n   SAGEMAKER_PROGRAM = inference.py\n   SAGEMAKER_SUBMODULE_DIRECTORY = /opt/ml/model/code\n")
print("="*70)
