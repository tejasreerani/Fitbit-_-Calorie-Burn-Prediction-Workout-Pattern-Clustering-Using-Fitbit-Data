# 🏋️‍♀️Fitbit-_-Calorie-Burn-Prediction-Workout-Pattern-Clustering-Using-Fitbit-Data
Data‑Driven Insights for Personalized Fitness

## 📌 Project Overview
Accurate calorie estimation is a critical feature in fitness apps.  
This project combines **supervised regression (KNN Regressor)** and **unsupervised clustering (KMeans + PCA)** to:
- Predict calories burned per workout session
- Discover hidden workout patterns and user segments

Built with **Streamlit** for interactive visualization.


## 🚀 Features
- 📊 Interactive dashboard with feature distributions, silhouette scores, and PCA cluster plots
- 🔥 Calorie burn prediction using trained KNN model
- 🏋️ Input dashboard for personalized fitness profiles
- 🎨 Clean UI with sidebar navigation and styled components


## 🛠 Tech Stack
- Python, Pandas, NumPy
- Scikit‑learn (Regression, Clustering, PCA)
- Streamlit (UI & deployment)
- Matplotlib / Seaborn (visualizations)

## 📂 Dataset
The dataset includes:
- Demographics (Age, Gender, Weight, Height, BMI)
- Physiological signals (Heart Rate, VO2Max, Resting HR)
- Workout details (Duration, Type, Frequency, Intensity)
- Hydration, Stress, Sleep
- Target: **Calories Burned**


## ⚙️ Installation

https://github.com/tejasreerani/Fitbit-_-Calorie-Burn-Prediction-Workout-Pattern-Clustering-Using-Fitbit-Data.git

cd Fitbit-CalorieBurn-Clustering

pip install -r requirements.txt

## 📂 Repository Structure
Fitbit-CalorieBurn-Clustering/
│
├── app.py                  # Streamlit code
├── knn_model.pkl           # Trained KNN model
├── Fitbit_dataset.csv      # Dataset (or link if too large)
├── README.md               # Project overview
├── /images                 # Screenshots

## 📊 Dashboard Preview
Home page
<img src="https://github.com/user-attachments/assets/cd01e836-7664-4030-bf00-71f38b0586b8" width="947" height="496" />

## Dashboard page
<img src="https://github.com/user-attachments/assets/6eec4172-f745-41b3-8fc6-652c4de47dc9" width="806" height="473" />

## Prediction page
<img src="https://github.com/user-attachments/assets/c89ca0c5-729f-42db-82b9-1365796fe63c" width="814" height="482" />

## 📦 Requirements

streamlit==1.32.0

pandas==2.2.1

numpy==1.26.4

matplotlib==3.8.3

seaborn==0.13.2

scikit-learn==1.4.1.post1

## 👩‍💻 Author
Kurmapu Lakshmi Tejasree

## 🔑 Conclusion
This project integrates calorie burn prediction with workout pattern clustering using Fitbit data. By combining supervised regression (KNN) and unsupervised learning (KMeans + PCA), it delivers accurate predictions and meaningful behavioral insights. Packaged with Streamlit for interactivity and GitHub for reproducibility, it stands out as a recruiter‑ready showcase of end‑to‑end data science.






