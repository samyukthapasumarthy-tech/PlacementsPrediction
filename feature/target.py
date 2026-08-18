import os
import pandas as pd


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
# 4. TARGET COLUMN
# --------------------------------------------------

target_column = "PlacementStatus"


# --------------------------------------------------
# 5. FEATURES FOR TARGET ENCODING
# --------------------------------------------------

target_features = [
    "Gender",
    "City",
    "Stream",
    "Specialisation"
]


# --------------------------------------------------
# 6. TAKE FIRST 10 ROWS
# --------------------------------------------------

data_10 = df[
    target_features + [target_column]
].head(10).copy()

print("\n==============================")
print("ORIGINAL FIRST 10 ROWS")
print("==============================")

print(data_10)


# --------------------------------------------------
# 7. PERFORM TARGET ENCODING
# --------------------------------------------------

encoded_df = data_10.copy()

for column in target_features:

    # Calculate mean target value
    # for each category using the complete dataset
    mean_encoding = df.groupby(
        column
    )[target_column].mean()

    # Replace category with mean target value
    encoded_df[column] = encoded_df[
        column
    ].map(mean_encoding)


# --------------------------------------------------
# 8. DISPLAY RESULT
# --------------------------------------------------

print("\n==============================")
print("TARGET ENCODED DATA")
print("==============================")

print(encoded_df)


# --------------------------------------------------
# 9. DISPLAY TARGET MEANS
# --------------------------------------------------

print("\n==============================")
print("TARGET ENCODING VALUES")
print("==============================")


for column in target_features:

    mean_encoding = df.groupby(
        column
    )[target_column].mean()

    print("\n", column)

    print(mean_encoding)


# --------------------------------------------------
# 10. OUTPUT DIRECTORY
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
# 11. SAVE ORIGINAL DATA
# --------------------------------------------------

data_10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_original_first_10.csv"
    ),
    index=False
)


# --------------------------------------------------
# 12. SAVE ENCODED DATA
# --------------------------------------------------

encoded_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_encoded_first_10.csv"
    ),
    index=False
)


# --------------------------------------------------
# 13. COMPLETION MESSAGE
# --------------------------------------------------

print("\n================================")
print("TARGET ENCODING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_DIR)