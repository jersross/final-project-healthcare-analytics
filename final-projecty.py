import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrameCollection
from awsgluedq.transforms import EvaluateDataQuality
from awsglue.dynamicframe import DynamicFrame
import re

# Script generated for node Custom Transform
def MyTransform(glueContext, dfc) -> DynamicFrameCollection:
    from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    df = dfc.select(list(dfc.keys())[0]).toDF()

    # 1. Conditionally fill missing Systolic BP & Diastolic BP using 'Blood Pressure (mmHg)' split
    split_bp = F.split(F.col("`Blood Pressure (mmHg)`"), "/")
    df = df.withColumn(
        "Systolic BP",
        F.coalesce(F.col("`Systolic BP`"), split_bp.getItem(0).cast("int")),
    ).withColumn(
        "Diastolic BP",
        F.coalesce(F.col("`Diastolic BP`"), split_bp.getItem(1).cast("int")),
    )
    df = df.drop("Blood Pressure (mmHg)")

    # 2. Impute missing Age first (by Sex median) so age_band is never NULL
    sex_window = Window.partitionBy("Sex").rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )
    age_median = F.expr("percentile_approx(Age, 0.5)").over(sex_window)
    df = df.withColumn("Age", F.coalesce(F.col("Age"), age_median))

    # 3. Derive age_band safely
    df = df.withColumn("age_band", (F.col("Age") / 10).cast("int") * 10)

    # 4. Define numerical columns requiring imputation
    impute_cols = [
        "Weight (kg)",
        "Height (m)",
        "Height (cm)",
        "BMI",
        "Abdominal Circumference (cm)",
        "Waist-to-Height Ratio",
        "Total Cholesterol (mg/dL)",
        "HDL (mg/dL)",
        "Fasting Blood Sugar (mg/dL)",
        "Estimated LDL (mg/dL)",
        "Systolic BP",
        "Diastolic BP",
        "CVD Risk Score",
    ]

    demo_window = Window.partitionBy("Sex", "age_band").rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )

    # Global window definition replaces .first()[0] driver collects
    global_window = Window.partitionBy().rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )

    for c in impute_cols:
        clean_name = (
            c.replace(" (", "_")
            .replace(")", "")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .lower()
        )
        flag_col = f"{clean_name}_was_imputed"
        df = df.withColumn(flag_col, F.col(c).isNull())

        # Step 1: Impute using Demographic Median (Sex + Age Band)
        group_median = F.expr(f"percentile_approx(`{c}`, 0.5)").over(demo_window)
        df = df.withColumn(c, F.coalesce(F.col(c), group_median))

        # Step 2: Distributed Fallback using Global Median (eliminates memory limit error)
        global_median = F.expr(f"percentile_approx(`{c}`, 0.5)").over(global_window)
        df = df.withColumn(c, F.coalesce(F.col(c), global_median))

    dyf = DynamicFrame.fromDF(df, glueContext, "imputed")
    return DynamicFrameCollection({"imputed": dyf}, glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Default ruleset used by all target nodes with data quality enabled
DEFAULT_DATA_QUALITY_RULESET = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Script generated for node Amazon S3
AmazonS3_node1786052771026 = glueContext.create_dynamic_frame.from_options(format_options={"quoteChar": "\"", "withHeader": True, "separator": ",", "optimizePerformance": False}, connection_type="s3", format="csv", connection_options={"paths": ["s3://mgmt-healthcare-analytics-jrms-final-project/patient-data/bronze/"]}, transformation_ctx="AmazonS3_node1786052771026")

# Script generated for node Change Schema
ChangeSchema_node1786052929801 = ApplyMapping.apply(frame=AmazonS3_node1786052771026, mappings=[("Sex", "string", "Sex", "string"), ("Age", "string", "Age", "int"), ("`Weight (kg)`", "string", "`Weight (kg)`", "double"), ("`Height (m)`", "string", "`Height (m)`", "double"), ("BMI", "string", "BMI", "double"), ("`Abdominal Circumference (cm)`", "string", "`Abdominal Circumference (cm)`", "double"), ("`Blood Pressure (mmHg)`", "string", "`Blood Pressure (mmHg)`", "int"), ("`Total Cholesterol (mg/dL)`", "string", "`Total Cholesterol (mg/dL)`", "double"), ("`HDL (mg/dL)`", "string", "`HDL (mg/dL)`", "int"), ("`Fasting Blood Sugar (mg/dL)`", "string", "`Fasting Blood Sugar (mg/dL)`", "int"), ("Smoking Status", "string", "Smoking Status", "string"), ("Diabetes Status", "string", "Diabetes Status", "string"), ("Physical Activity Level", "string", "Physical Activity Level", "string"), ("Family History of CVD", "string", "Family History of CVD", "string"), ("`Height (cm)`", "string", "`Height (cm)`", "int"), ("Waist-to-Height Ratio", "string", "Waist-to-Height Ratio", "double"), ("Systolic BP", "string", "Systolic BP", "int"), ("Diastolic BP", "string", "Diastolic BP", "long"), ("Blood Pressure Category", "string", "Blood Pressure Category", "string"), ("`Estimated LDL (mg/dL)`", "string", "`Estimated LDL (mg/dL)`", "int"), ("CVD Risk Score", "string", "CVD Risk Score", "double"), ("CVD Risk Level", "string", "CVD Risk Level", "string")], transformation_ctx="ChangeSchema_node1786052929801")

# Script generated for node Custom Transform
CustomTransform_node1786052977843 = MyTransform(glueContext, DynamicFrameCollection({"ChangeSchema_node1786052929801": ChangeSchema_node1786052929801}, glueContext))

# Script generated for node Select From Collection
SelectFromCollection_node1786057260361 = SelectFromCollection.apply(dfc=CustomTransform_node1786052977843, key=list(CustomTransform_node1786052977843.keys())[0], transformation_ctx="SelectFromCollection_node1786057260361")

# Script generated for node Evaluate Data Quality
EvaluateDataQuality_node1786056503865_ruleset = """
    Rules = [
        ColumnCount = 35,

        # Completeness Checks (Ensures 100% of Nulls were successfully imputed)
        IsComplete "Age",
        IsComplete "Sex",
        IsComplete "Weight (kg)",
        IsComplete "Height (m)",
        IsComplete "BMI",
        IsComplete "Abdominal Circumference (cm)",
        IsComplete "Waist-to-Height Ratio",
        IsComplete "Total Cholesterol (mg/dL)",
        IsComplete "HDL (mg/dL)",
        IsComplete "Fasting Blood Sugar (mg/dL)",
        IsComplete "Estimated LDL (mg/dL)",
        IsComplete "Systolic BP",
        IsComplete "Diastolic BP",
        IsComplete "CVD Risk Score",

        # Non-Zero Value Checks (Ensures valid physiological measurements)
        ColumnValues "Age" > 0,
        ColumnValues "Weight (kg)" > 0,
        ColumnValues "Height (m)" > 0,
        ColumnValues "BMI" > 0,
        ColumnValues "Abdominal Circumference (cm)" > 0,
        ColumnValues "Total Cholesterol (mg/dL)" > 0,
        ColumnValues "HDL (mg/dL)" > 0,
        ColumnValues "Fasting Blood Sugar (mg/dL)" > 0,
        ColumnValues "Systolic BP" >= 70,
        ColumnValues "Diastolic BP" >= 40
    ]
"""

EvaluateDataQuality_node1786056503865 = EvaluateDataQuality().process_rows(frame=SelectFromCollection_node1786057260361, ruleset=EvaluateDataQuality_node1786056503865_ruleset, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1786056503865", "enableDataQualityCloudWatchMetrics": True, "enableDataQualityResultsPublishing": True}, additional_options={"performanceTuning.caching":"CACHE_NOTHING"})

# Script generated for node ruleOutcomes
ruleOutcomes_node1786063849406 = SelectFromCollection.apply(dfc=EvaluateDataQuality_node1786056503865, key="ruleOutcomes", transformation_ctx="ruleOutcomes_node1786063849406")

# Script generated for node rowLevelOutcomes
rowLevelOutcomes_node1786063884303 = SelectFromCollection.apply(dfc=EvaluateDataQuality_node1786056503865, key="rowLevelOutcomes", transformation_ctx="rowLevelOutcomes_node1786063884303")

# Script generated for node keep_passed
keep_passed_node1786064069477 = Filter.apply(frame=rowLevelOutcomes_node1786063884303, f=lambda row: (bool(re.match("Passed", row["DataQualityEvaluationResult"]))), transformation_ctx="keep_passed_node1786064069477")

# Script generated for node keep_failed
keep_failed_node1786064043461 = Filter.apply(frame=rowLevelOutcomes_node1786063884303, f=lambda row: (bool(re.match("Failed", row["DataQualityEvaluationResult"]))), transformation_ctx="keep_failed_node1786064043461")

# Script generated for node Change Schema
ChangeSchema_node1786065218411 = ApplyMapping.apply(frame=keep_passed_node1786064069477, mappings=[("Sex", "string", "Sex", "string"), ("Age", "int", "Age", "int"), ("`Weight (kg)`", "double", "`Weight (kg)`", "double"), ("`Height (m)`", "double", "`Height (m)`", "double"), ("BMI", "double", "BMI", "double"), ("`Abdominal Circumference (cm)`", "double", "`Abdominal Circumference (cm)`", "double"), ("`Total Cholesterol (mg/dL)`", "double", "`Total Cholesterol (mg/dL)`", "double"), ("`HDL (mg/dL)`", "int", "`HDL (mg/dL)`", "int"), ("`Fasting Blood Sugar (mg/dL)`", "int", "`Fasting Blood Sugar (mg/dL)`", "int"), ("Smoking Status", "string", "Smoking Status", "string"), ("Diabetes Status", "string", "Diabetes Status", "string"), ("Physical Activity Level", "string", "Physical Activity Level", "string"), ("Family History of CVD", "string", "Family History of CVD", "string"), ("`Height (cm)`", "int", "`Height (cm)`", "int"), ("Waist-to-Height Ratio", "double", "Waist-to-Height Ratio", "double"), ("Systolic BP", "int", "Systolic BP", "int"), ("Diastolic BP", "long", "Diastolic BP", "long"), ("Blood Pressure Category", "string", "Blood Pressure Category", "string"), ("`Estimated LDL (mg/dL)`", "int", "`Estimated LDL (mg/dL)`", "int"), ("CVD Risk Score", "double", "CVD Risk Score", "double"), ("CVD Risk Level", "string", "CVD Risk Level", "string"), ("age_band", "int", "age_band", "int"), ("weight_kg_was_imputed", "boolean", "weight_kg_was_imputed", "boolean"), ("height_m_was_imputed", "boolean", "height_m_was_imputed", "boolean"), ("height_cm_was_imputed", "boolean", "height_cm_was_imputed", "boolean"), ("bmi_was_imputed", "boolean", "bmi_was_imputed", "boolean"), ("abdominal_circumference_cm_was_imputed", "boolean", "abdominal_circumference_cm_was_imputed", "boolean"), ("waist_to_height_ratio_was_imputed", "boolean", "waist_to_height_ratio_was_imputed", "boolean"), ("total_cholesterol_mg_dl_was_imputed", "boolean", "total_cholesterol_mg_dl_was_imputed", "boolean"), ("hdl_mg_dl_was_imputed", "boolean", "hdl_mg_dl_was_imputed", "boolean"), ("fasting_blood_sugar_mg_dl_was_imputed", "boolean", "fasting_blood_sugar_mg_dl_was_imputed", "boolean"), ("estimated_ldl_mg_dl_was_imputed", "boolean", "estimated_ldl_mg_dl_was_imputed", "boolean"), ("systolic_bp_was_imputed", "boolean", "systolic_bp_was_imputed", "boolean"), ("diastolic_bp_was_imputed", "boolean", "diastolic_bp_was_imputed", "boolean"), ("cvd_risk_score_was_imputed", "boolean", "cvd_risk_score_was_imputed", "boolean")], transformation_ctx="ChangeSchema_node1786065218411")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=keep_failed_node1786064043461, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1786052269908", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1786067190290 = glueContext.write_dynamic_frame.from_options(frame=keep_failed_node1786064043461, connection_type="s3", format="glueparquet", connection_options={"path": "s3://mgmt-healthcare-analytics-jrms-final-project/patient-data/rejected/", "partitionKeys": []}, format_options={"compression": "snappy"}, transformation_ctx="AmazonS3_node1786067190290")

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=ChangeSchema_node1786065218411, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1786052269908", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
AmazonS3_node1786065632257 = glueContext.getSink(path="s3://mgmt-healthcare-analytics-jrms-final-project/patient-data/silver/", connection_type="s3", updateBehavior="LOG", partitionKeys=[], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1786065632257")
AmazonS3_node1786065632257.setCatalogInfo(catalogDatabase="healthcare_db",catalogTableName="patient_data")
AmazonS3_node1786065632257.setFormat("glueparquet", compression="snappy")
AmazonS3_node1786065632257.writeFrame(ChangeSchema_node1786065218411)
job.commit()