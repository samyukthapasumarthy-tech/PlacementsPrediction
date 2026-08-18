import os
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


features = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]


data_10 = df[features].head(10).copy()

print("\nFirst 10 rows:")
print(data_10)

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(data_10)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=features
)

print("\n==============================")
print("STANDARDIZED DATA")
print("==============================")
print(standard_df)

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(data_10)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=features
)

print("\n==============================")
print("MIN-MAX SCALED DATA")
print("==============================")
print(minmax_df)


# --------------------------------------------------
# 7. ROBUST SCALING
# --------------------------------------------------

robust_scaler = RobustScaler()

robust_scaled = robust_scaler.fit_transform(data_10)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=features
)

print("\n==============================")
print("ROBUST SCALED DATA")
print("==============================")
print(robust_df)


# --------------------------------------------------
# 8. STANDARDIZATION MEAN AND STD
# --------------------------------------------------

print("\n==============================")
print("STANDARDIZED MEAN")
print("==============================")

print(standard_df.mean())


print("\n==============================")
print("STANDARDIZED STD")
print("==============================")

print(standard_df.std())


# --------------------------------------------------
# 9. STANDARD SCALER PARAMETERS
# --------------------------------------------------

print("\n==============================")
print("STANDARD SCALER MEAN")
print("==============================")

print(standard_scaler.mean_)


print("\n==============================")
print("STANDARD SCALER SCALE")
print("==============================")

print(standard_scaler.scale_)


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "feature_engineering"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


data_10.to_csv(
    os.path.join(OUTPUT_DIR, "original_first_10.csv"),
    index=False
)

standard_df.to_csv(
    os.path.join(OUTPUT_DIR, "standard_scaled_first_10.csv"),
    index=False
)

minmax_df.to_csv(
    os.path.join(OUTPUT_DIR, "minmax_scaled_first_10.csv"),
    index=False
)

robust_df.to_csv(
    os.path.join(OUTPUT_DIR, "robust_scaled_first_10.csv"),
    index=False
)


print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_DIR)