import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# --------------------------------------------------
# 1. FIND PROJECT DIRECTORY
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# --------------------------------------------------
# 2. DATA PATH
# --------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 4. ORDINAL FEATURE
# --------------------------------------------------

ordinal_feature = ["CGPA_Tier"]


# --------------------------------------------------
# 5. TAKE FIRST 10 ROWS
# --------------------------------------------------

data_10 = df[ordinal_feature].head(10).copy()

print("\n==============================")
print("ORIGINAL FIRST 10 ROWS")
print("==============================")

print(data_10)


# --------------------------------------------------
# 6. DEFINE ORDER
# --------------------------------------------------

categories = [
    ["Low", "Mid", "High"]
]


# --------------------------------------------------
# 7. CREATE ORDINAL ENCODER
# --------------------------------------------------

encoder = OrdinalEncoder(
    categories=categories
)


# --------------------------------------------------
# 8. PERFORM ORDINAL ENCODING
# --------------------------------------------------

encoded_data = encoder.fit_transform(
    data_10
)


# --------------------------------------------------
# 9. CONVERT TO DATAFRAME
# --------------------------------------------------

encoded_df = pd.DataFrame(
    encoded_data,
    columns=ordinal_feature
)


# --------------------------------------------------
# 10. DISPLAY RESULT
# --------------------------------------------------

print("\n==============================")
print("ORDINAL ENCODED DATA")
print("==============================")

print(encoded_df)


# --------------------------------------------------
# 11. DISPLAY MAPPING
# --------------------------------------------------

print("\n==============================")
print("ORDINAL MAPPING")
print("==============================")

print("Low  -> 0")
print("Mid  -> 1")
print("High -> 2")


# --------------------------------------------------
# 12. OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "feature_engineering"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# 13. SAVE RESULTS
# --------------------------------------------------

data_10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ordinal_original_first_10.csv"
    ),
    index=False
)

encoded_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ordinal_encoded_first_10.csv"
    ),
    index=False
)


# --------------------------------------------------
# 14. COMPLETION MESSAGE
# --------------------------------------------------

print("\n================================")
print("ORDINAL ENCODING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_DIR)