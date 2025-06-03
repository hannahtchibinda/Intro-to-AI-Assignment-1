# Step 1: import everything (some unused imports left in, just in case)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Step 2: upload the dataset to the left (I uploaded mine and it shows below)
# This line loads it in - make sure file name matches
df = pd.read_csv("/content/student_data.csv")

# Just checking it's loading properly
print("The first 5 rows of data:")
print(df.head())

# Step 3: Data Cleaning!
# Some entries are in the form 'varies' or 'unknown' which mess up the data
df.replace(["varies", "unknown"], np.nan, inplace=True)

# Making sure these columns are numbers, not strings
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df["study_hours_per_day"] = pd.to_numeric(df["study_hours_per_day"], errors='coerce')
df["mental_health_rating"] = pd.to_numeric(df["mental_health_rating"], errors='coerce')

# Get rid of rows with missing info
df.dropna(inplace=True)

# Step 4: Encode stuff (turn words into numbers for the model)
# Like gender, diet quality etc.
label_columns = [
    "gender", "part_time_job", "diet_quality", "parental_education_level",
    "internet_quality", "extracurricular_participation"
]

# quick loop to do all label encodings
for col in label_columns:
  encoder = LabelEncoder()
  df[col] = encoder.fit_transform(df[col])  # fits and transforms each one

# Step 5: Convert exam score into pass/fail
# (pass = 60 or more)
df["passed"] = (df["exam_score"] >= 60).astype(int)

# Step 6: Select the columns I want the model to learn from
features = [
    "age", "gender", "study_hours_per_day", "social_media_hours",
    "netflix_hours", "part_time_job", "attendance_percentage",
    "sleep_hours", "diet_quality", "exercise_frequency",
    "parental_education_level", "internet_quality",
    "mental_health_rating", "extracurricular_participation"
]

X = df[features]
y = df["passed"]

# Step 7: Split into training and testing sets
# (20% of data used for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Make the model!
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Test the model
y_pred = model.predict(X_test)

# Check how accurate it is
accuracy = accuracy_score(y_test, y_pred)
print(f"✔️Accuracy of the Model: {accuracy * 100:.2f}%")

# Final test print
print("\nSome real vs predicted values for reference:")
print(pd.DataFrame({
    "Actual Values:": y_test.values[:10],
    "Predicted Values:": y_pred[:10]
}))

