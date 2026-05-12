# TASK-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset

file_path = r"C:\Users\HAFIZ TECH\Downloads\diabetes.csv"   
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Dataset Information

print("\nDataset Info:")
df.info()

print("\nMissing Values (Initial):")
print(df.isnull().sum())

# 3. Treat Zero Values as Missing

zero_columns = [ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[zero_columns] = df[zero_columns].replace(0, np.nan)

print("\nMissing Values After Replacing Zeros:")
print(df.isnull().sum())

# 4. Handle Missing Values

for col in zero_columns:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Remove Duplicate Rows

duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before Removal:", duplicates_before)

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()
print("Duplicate Rows After Removal:", duplicates_after)

# 6. Descriptive Statistics

print("\nDescriptive Statistics:")
print(df.describe())

# 7. Outcome Distribution

print("\nOutcome Value Counts:")
print(df['Outcome'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

# 8. Distribution Plots

features = ['Glucose', 'BMI', 'Age', 'Insulin']

plt.figure(figsize=(12,8))
for i, col in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# 9. Boxplots (Outlier Detection)

plt.figure(figsize=(12,8))
for i, col in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# 10. Correlation Analysis

plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 11. Key Insights (Printed)

print("\nKEY INSIGHTS:")
print("1. Glucose has strong correlation with diabetes outcome.")
print("2. BMI and Age show moderate association with diabetes.")
print("3. Insulin contains outliers but carries useful information.")
print("4. Dataset is slightly imbalanced towards non-diabetic cases.")

# 12. Save Cleaned Dataset

df.to_csv("cleaned_diabetes.csv", index=False)
print("\n Cleaned dataset saved as 'cleaned_diabetes.csv'")

