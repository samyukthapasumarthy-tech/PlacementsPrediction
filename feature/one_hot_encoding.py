import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_PATH = r"C:\Users\samyu\Desktop\archive (2)\placement_predict_50k Dataset (2).csv"


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "PlacementStatus"


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}\n"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    return df


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ONE-HOT ENCODING + LOGISTIC REGRESSION")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    print(
        "\nDataset shape:",
        df.shape
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        df.head()
    )


    # --------------------------------------------------------
    # COLUMN NAMES
    # --------------------------------------------------------

    print(
        "\n\nDataset columns:"
    )

    for i, col in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:2}. {col}"
        )


    # --------------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"\nTarget column "
            f"'{TARGET_COLUMN}' "
            f"was not found."
        )


    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    print(
        "\n\nMISSING VALUE SUMMARY"
    )

    print("=" * 70)

    missing = (
        df.isnull()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:

        print(
            "No missing values found."
        )

    else:

        print(
            missing
        )


    # --------------------------------------------------------
    # REMOVE ROWS WHERE TARGET IS MISSING
    # --------------------------------------------------------

    target_missing = (
        df[TARGET_COLUMN]
        .isnull()
        .sum()
    )

    if target_missing > 0:

        print(
            f"\nRemoving "
            f"{target_missing} rows "
            f"with missing target."
        )

        df = df.dropna(
            subset=[TARGET_COLUMN]
        )


    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\n\nPLACEMENT STATUS DISTRIBUTION"
    )

    print("=" * 70)

    print(
        df[TARGET_COLUMN]
        .value_counts()
    )

    print(
        "\nPlacement percentages:"
    )

    print(
        (
            df[TARGET_COLUMN]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


    # ========================================================
    # 6. REMOVE COLUMNS THAT SHOULD NOT BE FEATURES
    # ========================================================

    # StudentID is only an identifier.
    #
    # Salary Package must NOT be used to predict placement
    # because salary is known after placement and can cause
    # target leakage.
    #
    # IsAnomaly is also excluded from the prediction model.

    columns_to_drop = [
        "StudentID",
        "Salary Package",
        "IsAnomaly"
    ]

    # Only remove columns that actually exist.

    columns_to_drop = [
        col
        for col in columns_to_drop
        if col in df.columns
    ]


    # --------------------------------------------------------
    # CREATE X AND y
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            TARGET_COLUMN
        ] + columns_to_drop
    )

    y = df[TARGET_COLUMN]


    print(
        "\n\nFEATURE INFORMATION"
    )

    print("=" * 70)

    print(
        "Target:",
        TARGET_COLUMN
    )

    print(
        "Dropped columns:",
        columns_to_drop
    )

    print(
        "Number of features before encoding:",
        X.shape[1]
    )


    # ========================================================
    # 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
    # ========================================================

    numerical_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )


    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )


    print(
        "\nNumerical features:"
    )

    print(
        numerical_features
    )


    print(
        "\nCategorical features:"
    )

    print(
        categorical_features
    )


    # ========================================================
    # 8. TRAIN / VALIDATION SPLIT
    # ========================================================

    X_train, X_val, y_train, y_val = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print(
        "\n\nDATA SPLIT"
    )

    print("=" * 70)

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )


    # ========================================================
    # 9. NUMERICAL PIPELINE
    # ========================================================

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # ========================================================
    # 10. CATEGORICAL PIPELINE
    # ========================================================

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # ========================================================
    # 11. COLUMN TRANSFORMER
    # ========================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    # ========================================================
    # 12. LOGISTIC REGRESSION MODEL
    # ========================================================

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )


    # ========================================================
    # 13. TRAIN MODEL
    # ========================================================

    print(
        "\n\nTRAINING LOGISTIC REGRESSION..."
    )

    model.fit(
        X_train,
        y_train
    )


    print(
        "Training completed."
    )


    # ========================================================
    # 14. VALIDATION PREDICTIONS
    # ========================================================

    y_pred = model.predict(
        X_val
    )


    # ========================================================
    # 15. ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_val,
        y_pred
    )


    print(
        "\n\nMODEL PERFORMANCE"
    )

    print("=" * 70)

    print(
        "Validation Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Validation Accuracy (%):",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )


    # ========================================================
    # 16. CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n\nCLASSIFICATION REPORT"
    )

    print("=" * 70)

    print(
        classification_report(
            y_val,
            y_pred
        )
    )


    # ========================================================
    # 17. CONFUSION MATRIX
    # ========================================================

    print(
        "\nCONFUSION MATRIX"
    )

    print("=" * 70)

    cm = confusion_matrix(
        y_val,
        y_pred
    )

    print(
        cm
    )


    # ========================================================
    # 18. SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ]
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )


    # ========================================================
    # 19. GET ENCODED FEATURE NAMES
    # ========================================================

    fitted_preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )


    try:

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        print(
            "\n\nENCODED FEATURE COUNT:"
        )

        print(
            len(feature_names)
        )

        print(
            "\nFirst 20 encoded features:"
        )

        for feature in (
            feature_names[:20]
        ):

            print(
                feature
            )

    except Exception:

        feature_names = None


    # ========================================================
    # 20. MODEL COEFFICIENTS
    # ========================================================

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    coefficients = (
        classifier.coef_[0]
    )


    if feature_names is not None:

        coefficient_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Coefficient":
                coefficients,

            "AbsoluteCoefficient":
                np.abs(
                    coefficients
                )
        })

        coefficient_df = (
            coefficient_df
            .sort_values(
                "AbsoluteCoefficient",
                ascending=False
            )
        )


        print(
            "\n\nTOP FEATURES"
        )

        print("=" * 70)

        print(
            coefficient_df[
                [
                    "Feature",
                    "Coefficient"
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
        )


        coefficient_path = os.path.join(
            OUTPUT_DIR,
            "logistic_coefficients.csv"
        )

        coefficient_df.to_csv(
            coefficient_path,
            index=False
        )


    # ========================================================
    # 21. PREDICT A NEW STUDENT
    # ========================================================

    print(
        "\n\nNEW STUDENT PREDICTION"
    )

    print("=" * 70)


    # Create a new student using the
    # actual columns present in X.

    new_student = {}

    for col in X.columns:

        if col in [
            "CGPA",
            "AptitudeTestScore",
            "CodingTestScore",
            "MockInterviewScore",
            "AttendancePercent"
        ]:

            # Example values

            example_values = {

                "CGPA": 8.5,

                "AptitudeTestScore": 82,

                "CodingTestScore": 85,

                "MockInterviewScore": 78,

                "AttendancePercent": 90
            }

            new_student[col] = [
                example_values[col]
            ]

        elif col in numerical_features:

            # Use median for other numerical
            # columns

            new_student[col] = [
                X_train[col].median()
            ]

        elif col in categorical_features:

            # Use most common category
            # from training data

            new_student[col] = [
                X_train[col]
                .mode()[0]
            ]


    new_student_df = pd.DataFrame(
        new_student
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    new_prediction = model.predict(
        new_student_df
    )


    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    new_probability = (
        model.predict_proba(
            new_student_df
        )
    )


    if new_prediction[0] == 1:

        print(
            "Prediction: PLACED"
        )

    else:

        print(
            "Prediction: NOT PLACED"
        )


    print(
        "\nProbability of NOT PLACED:",
        round(
            new_probability[0][0],
            4
        )
    )


    print(
        "Probability of PLACED:",
        round(
            new_probability[0][1],
            4
        )
    )


    # ========================================================
    # 22. SAVE VALIDATION PREDICTIONS
    # ========================================================

    prediction_output = X_val.copy()

    prediction_output[
        "ActualPlacement"
    ] = y_val.values

    prediction_output[
        "PredictedPlacement"
    ] = y_pred

    prediction_output[
        "PlacementProbability"
    ] = model.predict_proba(
        X_val
    )[:, 1]


    predictions_path = os.path.join(
        OUTPUT_DIR,
        "validation_predictions.csv"
    )

    prediction_output.to_csv(
        predictions_path,
        index=False
    )


    # ========================================================
    # 23. SAVE SUMMARY REPORT
    # ========================================================

    report_path = os.path.join(
        OUTPUT_DIR,
        "logistic_regression_report.txt"
    )


    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "LOGISTIC REGRESSION REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Dataset shape: {df.shape}\n"
        )

        f.write(
            f"Training samples: "
            f"{len(X_train)}\n"
        )

        f.write(
            f"Validation samples: "
            f"{len(X_val)}\n\n"
        )

        f.write(
            f"Target: "
            f"{TARGET_COLUMN}\n"
        )

        f.write(
            f"Dropped columns: "
            f"{columns_to_drop}\n\n"
        )

        f.write(
            f"Numerical features: "
            f"{numerical_features}\n\n"
        )

        f.write(
            f"Categorical features: "
            f"{categorical_features}\n\n"
        )

        f.write(
            f"Validation Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Validation Accuracy (%): "
            f"{accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
            + "\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(
            classification_report(
                y_val,
                y_pred
            )
        )

        f.write(
            "\n\nNew Student Prediction: "
        )

        f.write(
            "PLACED"
            if new_prediction[0] == 1
            else "NOT PLACED"
        )

        f.write(
            "\n"
        )

        f.write(
            f"Probability of Placed: "
            f"{new_probability[0][1]:.4f}\n"
        )


    # ========================================================
    # 24. FINAL OUTPUT
    # ========================================================

    print(
        "\n\nFILES SAVED"
    )

    print("=" * 70)

    print(
        "Confusion matrix:"
    )

    print(
        cm_path
    )

    if feature_names is not None:

        print(
            "\nLogistic coefficients:"
        )

        print(
            coefficient_path
        )

    print(
        "\nValidation predictions:"
    )

    print(
        predictions_path
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROGRAM COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )