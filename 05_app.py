import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.title("🩺 Diabetes Prediction System")

uploaded_file = st.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    # -----------------------------
    # Data Cleaning
    # -----------------------------

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    st.success("Data cleaned successfully.")

    st.subheader("Clean Dataset")
    st.dataframe(df)

    # -----------------------------
    # Model Training
    # -----------------------------

    target = "diabetes"

    if target in df.columns:

        X = df.drop(target, axis=1)
        y = df[target]

        categorical_features = X.select_dtypes(
            include=["object"]
        ).columns

        numerical_features = X.select_dtypes(
            exclude=["object"]
        ).columns

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        acc = accuracy_score(y_test, prediction)

        st.subheader("Model Accuracy")

        st.metric("Accuracy", f"{acc*100:.2f}%")

        st.divider()

        st.subheader("Predict New Patient")

        user_input = {}

        for col in X.columns:

            if col in numerical_features:
                user_input[col] = st.number_input(
                    col,
                    value=float(X[col].median())
                )
            else:
                user_input[col] = st.selectbox(
                    col,
                    X[col].unique()
                )

        if st.button("Predict"):

            input_df = pd.DataFrame([user_input])

            pred = model.predict(input_df)[0]

            if pred == 1:
                st.error("⚠️ Patient is likely to have Diabetes")
            else:
                st.success("✅ Patient is unlikely to have Diabetes")

    else:
        st.error("Target column 'diabetes' not found.")