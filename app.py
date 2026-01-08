import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("📊 Customer Churn Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/churn_data.csv")

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Preprocessing
data = data.dropna()

label_encoder = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = label_encoder.fit_transform(data[col])

X = data.drop("Churn", axis=1)
y = data["Churn"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.success(f"Accuracy: {round(accuracy * 100, 2)}%")

st.subheader("🔮 Predict Churn for New Customer")

input_data = []
for col in data.drop("Churn", axis=1).columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    result = "Churn" if prediction == 1 else "Not Churn"
    st.warning(f"Prediction: **{result}**")
