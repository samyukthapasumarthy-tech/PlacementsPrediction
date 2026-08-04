import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(BASE_DIR, "..", "Data", "placement_predict_Dataset (2).csv")
#
# report_path = os.path.join(BASE_DIR, "..", "Data", "placement_predict_Dataset (2).csv")

# loading dataset
df=pd.read_csv(r"C:\Users\samyu\Desktop\placement_predict_50k Dataset (2).csv")
# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["CGPA"], bins=10, edgecolor="black")
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.show()

# Pie Chart
df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x="PlacementStatus", y="CGPA", data=df)
plt.title("CGPA vs Placement Status")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="CGPA", y="AttendancePercent", data=df)
plt.title("CGPA vs Attendance Percentage")
plt.show()
print("Shape of the Dataset:", df.shape)

# Number of rows and columns separately
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
df["Gender"].value_counts().plot(kind="bar")

plt.title("Frequency Distribution of Gender")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()
# Select only numerical columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Plot heat map
plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,        # Display correlation values
    cmap="coolwarm",   # Color scheme
    linewidths=0.5
)

plt.title("Correlation Heat Map")
plt.show()