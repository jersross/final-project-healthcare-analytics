# Install required libraries inside Google Colab
!pip install -q gradio pandas scikit-learn seaborn matplotlib pyarrow boto3 scipy

import os
import io
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import gradio as gr
import boto3

# ==========================================================
# 1. DATA INGESTION (PARQUET/CSV FALLBACK)
# ==========================================================
PARQUET_FILE = "run-1786068036316-part-block-0-r-00000-snappy (2).parquet"
CSV_FILE = "CVD Dataset (1).csv"

def load_dataset():
    if os.path.exists(PARQUET_FILE):
        print(f"Loading local Parquet file: {PARQUET_FILE}")
        return pd.read_parquet(PARQUET_FILE)
    elif os.path.exists(CSV_FILE):
        print(f"Loading local CSV file: {CSV_FILE}")
        return pd.read_csv(CSV_FILE)
    else:
        try:
            from google.colab import files
            print("No local file found. Please upload dataset below:")
            uploaded = files.upload()
            fname = list(uploaded.keys())[0]
            if fname.endswith('.parquet'):
                return pd.read_parquet(io.BytesIO(uploaded[fname]))
            else:
                return pd.read_csv(io.BytesIO(uploaded[fname]))
        except Exception as e:
            raise FileNotFoundError(f"Could not load data. Ensure dataset is uploaded. Error: {e}")

df_raw = load_dataset()

# ==========================================================
# 2. ALL 35 PARQUET SCHEMA FIELDS DEFINITION
# ==========================================================
demographic_cols = ['Sex', 'Age']
anthropometric_cols = ['Weight (kg)', 'Height (m)', 'Height (cm)', 'BMI', 'Abdominal Circumference (cm)', 'Waist-to-Height Ratio']
vitals_cols = ['Systolic BP', 'Diastolic BP', 'Blood Pressure Category']
biochemical_cols = ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)']
lifestyle_cols = ['Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
target_cols = ['CVD Risk Level', 'CVD Risk Score']
derived_cols = ['age_band']
flag_cols = [c for c in df_raw.columns if c.endswith('_was_imputed')]

df = df_raw.copy()

# ==========================================================
# 3. LOCAL MODEL PIPELINE (FALLBACK / IN-MEMORY)
# ==========================================================
cat_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
num_features = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)']

label_encoders = {}
df_model = df.copy()
for col in cat_features:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

target_le = LabelEncoder()
df_model['CVD_Risk_Level_Encoded'] = target_le.fit_transform(df_model['CVD Risk Level'])

feature_cols = num_features + cat_features
X = df_model[feature_cols]
y_class = df_model['CVD_Risk_Level_Encoded']
y_reg = df_model['CVD Risk Score']

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Stage 1 Classifier
clf = HistGradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_class_train)

# Stage 2 Regressor
X_train_reg = X_train.copy()
X_train_reg['CVD_Risk_Level_Signal'] = y_class_train
X_test_reg = X_test.copy()
X_test_reg['CVD_Risk_Level_Signal'] = clf.predict(X_test)

reg = HistGradientBoostingRegressor(random_state=42)
reg.fit(X_train_reg, y_reg_train)

# Unsupervised K-Means
cluster_cols = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 'Fasting Blood Sugar (mg/dL)']
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_cols].fillna(df[cluster_cols].median()))
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster_Segment'] = kmeans.fit_predict(X_cluster_scaled)

# ==========================================================
# 4. GRADIO INTERFACE BUILD
# ==========================================================
with gr.Blocks(title="Cardiovascular Disease (CVD) Analytics & Decision System") as demo:
    gr.Markdown("# 🏥 CVD Risk Stratification & Clinical Analytics Portal")
    gr.Markdown("Connected to AWS S3 Data Lake (Medallion Architecture) and SageMaker Inference Endpoint.")

    with gr.Tab("Executive Overview & Lineage"):
        gr.Markdown("### 📊 Pipeline Health & Data Lineage Audit")
        with gr.Row():
            gr.Metric("Total Screened Records", f"{len(df):,}")
            gr.Metric("Average Patient Risk Score", f"{df['CVD Risk Score'].mean():.2f}")
            gr.Metric("High Risk Prevalence", f"{(df['CVD Risk Level']=='HIGH').mean()*100:.1f}%")

        gr.Markdown("#### Data Quality Imputation Audit (AWS Glue ETL Flags)")
        if flag_cols:
            impute_summary = df[flag_cols].mean().reset_index()
            impute_summary.columns = ['Imputed Attribute Flag', 'Imputation Rate (%)']
            impute_summary['Imputation Rate (%)'] = (impute_summary['Imputation Rate (%)'] * 100).round(2)
            gr.DataFrame(impute_summary)
        else:
            gr.Markdown("_No Glue imputation flags detected in current dataset view._")

    with gr.Tab("Descriptive Exploration"):
        gr.Markdown("### 📈 Biometric Correlations")
        def plot_corr():
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_features + ['CVD Risk Score']].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            plt.title("Pearson Correlation Matrix")
            plt.tight_layout()
            return fig
        gr.Plot(plot_corr)

    with gr.Tab("Patient Profiling (K-Means)"):
        gr.Markdown("### 🧬 Unsupervised Clinical Cohorts")
        def eval_clusters(k_val):
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels = km.fit_predict(X_cluster_scaled)
            sil = silhouette_score(X_cluster_scaled, labels)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(x=df['Age'], y=df['BMI'], hue=labels, palette='tab10', ax=ax)
            plt.title(f"Patient Segmentation (k={k_val}) - Silhouette Score: {sil:.3f}")
            return fig
        k_slider = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Select Cluster Count (k)")
        cluster_plot = gr.Plot()
        k_slider.change(eval_clusters, inputs=k_slider, outputs=cluster_plot)

    with gr.Tab("Predictive Analytics"):
        gr.Markdown("### 🎯 Model Performance & Hypothesis Testing")
        preds = reg.predict(X_test_reg)
        r2 = r2_score(y_reg_test, preds)
        mae = mean_absolute_error(y_reg_test, preds)
        rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
        
        gr.Markdown(f"**Test Set R² Score:** {r2:.3f} | **MAE:** {mae:.2f} | **RMSE:** {rmse:.2f}")

        def plot_hypothesis():
            smokers = df[df['Smoking Status']=='Y']['CVD Risk Score']
            non_smokers = df[df['Smoking Status']=='N']['CVD Risk Score']
            t_stat, p_val = stats.ttest_ind(smokers, non_smokers, equal_var=False)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.boxplot(x='Smoking Status', y='CVD Risk Score', data=df, ax=ax, palette='Set2')
            plt.title(f"Smoking vs. Risk Score (Welch's t-test p-val: {p_val:.4e})")
            return fig
        gr.Plot(plot_hypothesis)

    with gr.Tab("Patient Simulator (Inference)"):
        gr.Markdown("### 🩺 Real-Time Clinical Risk Score Assessment")
        
        # AWS Integration Credentials Placeholders
        with gr.Accordion("AWS SageMaker Endpoint Configuration (Optional)", open=False):
            aws_key = gr.Textbox(label="AWS Access Key ID", type="password", placeholder="AKIA...")
            aws_secret = gr.Textbox(label="AWS Secret Access Key", type="password", placeholder="wJalrX...")
            aws_region = gr.Textbox(value="us-east-1", label="AWS Region")
            endpoint_name = gr.Textbox(value="cvd-risk-prediction-endpoint", label="SageMaker Endpoint Name")

        with gr.Row():
            age_in = gr.Number(value=55, label="Age")
            sex_in = gr.Dropdown(choices=["M", "F"], value="M", label="Sex")
            bmi_in = gr.Number(value=28.4, label="BMI")
            sys_in = gr.Number(value=138, label="Systolic BP")
            dia_in = gr.Number(value=88, label="Diastolic BP")
        with gr.Row():
            chol_in = gr.Number(value=215, label="Total Cholesterol (mg/dL)")
            hdl_in = gr.Number(value=42, label="HDL (mg/dL)")
            fbs_in = gr.Number(value=110, label="Fasting Blood Sugar (mg/dL)")
            smoke_in = gr.Dropdown(choices=["Y", "N"], value="N", label="Smoking Status")
            diab_in = gr.Dropdown(choices=["Y", "N"], value="N", label="Diabetes Status")
            act_in = gr.Dropdown(choices=["Low", "Moderate", "High"], value="Moderate", label="Physical Activity")
            fam_in = gr.Dropdown(choices=["Y", "N"], value="N", label="Family History of CVD")

        calc_btn = gr.Button("Calculate Patient CVD Risk", variant="primary")
        
        out_cluster = gr.Textbox(label="Assigned Cluster Cohort")
        out_risk_level = gr.Textbox(label="Predicted CVD Risk Level (Stage 1)")
        out_risk_score = gr.Number(label="Predicted Quantified CVD Risk Score (Stage 2)")

        def run_inference(age, sex, bmi, sys_bp, dia_bp, chol, hdl, fbs, smoke, diab, act, fam, key, secret, region, ep_name):
            # Cluster assignment
            sample_scaled = scaler_cluster.transform([[age, bmi, sys_bp, dia_bp, chol, fbs]])
            cluster_id = kmeans.predict(sample_scaled)[0]

            # Try AWS SageMaker Endpoint if keys provided
            if key and secret and ep_name:
                try:
                    client = boto3.client(
                        'sagemaker-runtime',
                        region_name=region,
                        aws_access_key_id=key,
                        aws_secret_access_key=secret
                    )
                    payload = {
                        "Age": age, "Sex": sex, "BMI": bmi, "Systolic BP": sys_bp, "Diastolic BP": dia_bp,
                        "Total Cholesterol (mg/dL)": chol, "HDL (mg/dL)": hdl, "Fasting Blood Sugar (mg/dL)": fbs,
                        "Smoking Status": smoke, "Diabetes Status": diab,
                        "Physical Activity Level": act, "Family History of CVD": fam
                    }
                    response = client.invoke_endpoint(
                        EndpointName=ep_name,
                        ContentType='application/json',
                        Body=json.dumps(payload)
                    )
                    res = json.loads(response['Body'].read().decode())
                    return f"Cluster {cluster_id} (AWS)", res['predicted_risk_level'], res['predicted_risk_score']
                except Exception as e:
                    print(f"SageMaker Endpoint call failed, using local fallback. Error: {e}")

            # Local Fallback Execution
            input_dict = {
                'Age': age, 'BMI': bmi, 'Systolic BP': sys_bp, 'Diastolic BP': dia_bp,
                'Total Cholesterol (mg/dL)': chol, 'HDL (mg/dL)': hdl, 'Fasting Blood Sugar (mg/dL)': fbs,
                'Sex': label_encoders['Sex'].transform([sex])[0],
                'Smoking Status': label_encoders['Smoking Status'].transform([smoke])[0],
                'Diabetes Status': label_encoders['Diabetes Status'].transform([diab])[0],
                'Physical Activity Level': label_encoders['Physical Activity Level'].transform([act])[0],
                'Family History of CVD': label_encoders['Family History of CVD'].transform([fam])[0]
            }
            input_df = pd.DataFrame([input_dict])[feature_cols]
            pred_class_enc = clf.predict(input_df)[0]
            pred_class_str = target_le.inverse_transform([pred_class_enc])[0]

            input_df_reg = input_df.copy()
            input_df_reg['CVD_Risk_Level_Signal'] = pred_class_enc
            pred_score = reg.predict(input_df_reg)[0]

            return f"Cluster {cluster_id} (Local)", pred_class_str, round(float(pred_score), 2)

        calc_btn.click(
            run_inference,
            inputs=[age_in, sex_in, bmi_in, sys_in, dia_in, chol_in, hdl_in, fbs_in, smoke_in, diab_in, act_in, fam_in, aws_key, aws_secret, aws_region, endpoint_name],
            outputs=[out_cluster, out_risk_level, out_risk_score]
        )

demo.launch(share=True)