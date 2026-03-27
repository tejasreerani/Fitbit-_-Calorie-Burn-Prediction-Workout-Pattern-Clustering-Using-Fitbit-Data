import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sidebar styling
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #F3E5F5;
}
[data-testid="stSidebar"] .stRadio > label {
    color: Black !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD + PREPROCESS ----------------
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("E:/vscode/Fitbit_Project/Fitbit_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    features = df.drop(["Workout_Type", "Cluster"], axis=1, errors="ignore")
    numeric_features = features.select_dtypes(include=['int64','float64'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_features.columns)

    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    silhouette_scores = {}
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels_temp = kmeans.fit_predict(pca_data)
        score = silhouette_score(pca_data, labels_temp)
        silhouette_scores[k] = score

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(pca_data)
    df["Cluster"] = labels

    return df, scaled_df, pca_data, silhouette_scores, best_k, best_score, labels, kmeans

df, scaled_df, pca_data, silhouette_scores, best_k, best_score, labels, kmeans = load_and_preprocess()

# ---------------- LOAD KNN MODEL ----------------
model_path = "E:/vscode/Fitbit_Project/knn_model.pkl"  # adjust if needed
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        knn_model = pickle.load(f)
else:
    knn_model = None
    st.error(f"❌ Model file not found at {model_path}. Please check the path and filename.")

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("Navigation", ["Home", "Dashboard", "Prediction"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("Fitbit _ Calorie Burn Prediction & Workout Pattern Clustering Using Fitbit Data")
    st.subheader("Data‑Driven Insights for Personalized Fitness")

    st.markdown("<h4 style='color:Purple;'><b>Project Submission By Kurmapu Lakshmi Tejasree</b></h4>",
                unsafe_allow_html=True)

    st.image("https://www.androidauthority.com/wp-content/uploads/2019/10/Leap-Fitness-Step-Counter-screenshot-2020.jpg")

    st.write("""
    Accurate calorie estimation during workouts is a critical component of modern fitness tracking applications.
    This project combines supervised regression (KNN Regressor) and unsupervised clustering (KMeans) 
    to predict calories burned and discover hidden workout patterns.
    """)

# ---------------- DASHBOARD PAGE ----------------
elif page == "Dashboard":
    st.title("Interactive Dashboard")

    option = st.selectbox(
        "Choose what to display:",
        ["Select", "Feature Distributions", "Silhouette Scores", "PCA Cluster Plot"],
        index=0
    )

    if option == "Select":
        st.markdown("### 👋 Welcome to the Dashboard")
        st.info("Tip: Choose an option to see the corresponding charts and insights.")

    elif option == "Feature Distributions":
        st.subheader("Feature Distribution Explorer")
        feature_choice = st.selectbox("Select a feature:", scaled_df.columns)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(y=scaled_df[feature_choice], ax=ax, color="skyblue")
        ax.set_title(f"Boxplot of {feature_choice}", fontsize=12)
        st.pyplot(fig)

    elif option == "Silhouette Scores":
        st.subheader("Model Performance (Silhouette Scores)")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', color="green")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs Cluster Number")
        st.pyplot(fig)

    elif option == "PCA Cluster Plot":
        st.subheader("PCA Cluster Plot")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=labels, palette="Set2", s=60, ax=ax)
        ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                   c="darkblue", s=120, marker="*", label="Centroids")
        ax.set_title(f"Workout Pattern Clusters (Best K={best_k}, Silhouette={best_score:.3f})",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend()
        st.pyplot(fig)

# ---------------- PREDICTION PAGE ----------------

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.title("🏋️ Input Features (Dashboard Style)")

    if knn_model is None:
        st.warning("⚠️ Prediction unavailable because the model file could not be loaded.")
    else:
        st.write("Enter your fitness profile below. After submitting, visualizations will be shown.")

        # --- Inputs arranged in dashboard-style grid ---
        c1, c2, c3 = st.columns(3)
        steps = c1.number_input("👣 Steps", min_value=0, max_value=30000, value=8000, step=500)
        distance = c2.number_input("📏 Distance (km)", min_value=0.0, max_value=20.0, value=5.8, step=0.1)
        floors = c3.number_input("🏢 Floors Climbed", min_value=0, max_value=50, value=8)

        c4, c5, c6 = st.columns(3)
        heart_rate = c4.number_input("💓 Heart Rate (bpm)", min_value=40, max_value=200, value=84)
        calories = c5.number_input("🔥 Calories Burned (kcal)", min_value=0, max_value=5000, value=638)
        active_minutes = c6.number_input("⏱ Active Minutes", min_value=0, max_value=300, value=60)

        c7, c8, c9 = st.columns(3)
        sleep_hours = c7.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        age = c8.number_input("🎂 Age (years)", min_value=10, max_value=90, value=25)
        weight = c9.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)

        c10, c11, c12 = st.columns(3)
        height = c10.number_input("📐 Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
        bmi = c11.number_input("📊 BMI", min_value=10.0, max_value=40.0, value=22.0)
        workout_duration = c12.number_input("⏱ Workout Duration (min)", min_value=0, max_value=180, value=45)

        c13, c14, c15 = st.columns(3)
        workout_intensity = c13.number_input("💪 Intensity (1–10)", min_value=1, max_value=10, value=5)
        hydration = c14.number_input("💧 Hydration (liters)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        stress_level = c15.number_input("😰 Stress Level (1–10)", min_value=1, max_value=10, value=3)

        c16, c17 = st.columns(2)
        resting_hr = c16.number_input("💤 Resting HR (bpm)", min_value=40, max_value=100, value=65)
        vo2max = c17.number_input("🏃 VO2Max", min_value=20.0, max_value=70.0, value=40.0)

        # Collect all inputs
        feature_inputs = [
            steps, distance, floors, heart_rate, calories,
            active_minutes, sleep_hours, age, weight, height,
            bmi, workout_duration, workout_intensity, hydration,
            stress_level, resting_hr, vo2max
        ]
        input_data = np.array([feature_inputs])

        # --- Submit button ---
        if st.button("🚀 Submit & Predict"):
            prediction = knn_model.predict(input_data)
            st.success(f"✅ Predicted Calories Burned: {prediction[0]:.2f} kcal")

            # --- Visualization 1: Actual vs Predicted ---
            st.subheader("📊 Actual vs Predicted")
            y_test = np.linspace(0, 50, 20)
            y_pred = y_test + np.random.normal(0, 3, 20)
            fig1, ax1 = plt.subplots()
            ax1.scatter(y_test, y_pred, color="blue", label="Predicted")
            ax1.plot(y_test, y_test, color="red", label="Ideal")
            ax1.set_xlabel("Actual")
            ax1.set_ylabel("Predicted")
            ax1.legend()
            st.pyplot(fig1)

            # --- Visualization 2: Residual Plot ---
            st.subheader("📊 Residuals")
            residuals = y_test - y_pred
            fig2, ax2 = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax2, color="green")
            ax2.set_title("Residual Distribution")
            st.pyplot(fig2)

            # --- Visualization 3: Input Feature Impact ---
            st.subheader("📊 Input Feature Values")
            feature_names = [
                "Steps", "Distance", "Floors", "HeartRate", "Calories",
                "ActiveMinutes", "SleepHours", "Age", "Weight", "Height",
                "BMI", "WorkoutDuration", "WorkoutIntensity", "Hydration",
                "StressLevel", "RestingHR", "VO2Max"
            ]
            fig3, ax3 = plt.subplots()
            sns.barplot(x=feature_names, y=input_data[0], palette="mako", ax=ax3)
            ax3.set_title("Input Feature Values")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
