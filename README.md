Dashboard link - https://frauddetectionmodeldeeplearning-kxs2zxcx5jb3slmfllqgzg.streamlit.app/

# Fraud Detection Model  

## Project Details  
This project focuses on building a **fraud detection model** using a **neural network** implemented with **PyTorch** and understanding the effect of hyperparameter tuning on model performance.  
The primary goal is to classify financial transactions as **fraudulent or non-fraudulent** based on transaction-related features.  

### Key Activities  
- **Data Preprocessing:** Encoding categorical variables, scaling numerical variables.  
- **Model Development:** Custom neural network architecture for classification.  
- **Hyperparameter Tuning:** Configurable parameters via **Streamlit UI**.  
- **Model Evaluation:** Accuracy, confusion matrix, ROC curve, precision-recall curve.  
- **Feature Importance Analysis:** Using **Permutation Importance** to determine key influencing factors.  

---

## Technologies Used  
- **Python** – Programming language.  
- **PyTorch** – Deep learning framework for neural networks.  
- **Scikit-learn** – Data preprocessing, model evaluation, feature importance analysis.  
- **Pandas & NumPy** – Data manipulation and numerical processing.  
- **Streamlit** – Web application interface for model training and visualization.  
- **Matplotlib & Seaborn** – Data visualization for insights and performance metrics.  

---

## Nature of Data  

### Categorical Variables (Encoded)  
- **Transaction_Type** – Online, POS, ATM, etc.  
- **Device_Type** – Mobile, laptop, desktop.  
- **Location** – Geographical location of the transaction.  
- **Merchant_Category** – Merchant type.  
- **IP_Address_Flag** – Indicates if the IP is flagged.  
- **Previous_Fraudulent_Activity** – Customer’s past fraud history.  
- **Card_Type** – Credit or debit card type.  
- **Authentication_Method** – OTP, biometric, PIN.  
- **Is_Weekend** – Whether the transaction occurred on a weekend.  

### Numerical Variables (Scaled)  
- **Transaction_Amount** – Value of the transaction.  
- **Account_Balance** – Account balance before the transaction.  
- **Daily_Transaction_Count** – Number of transactions per day.  
- **Avg_Transaction_Amount_7d** – Average transaction amount over 7 days.  
- **Failed_Transaction_Count_7d** – Number of failed transactions in 7 days.  
- **Card_Age** – Age of the card used.  
- **Transaction_Distance** – Distance between transaction location and home.  
- **Risk_Score** – System-generated risk score.  

### Target Variable  
- **Fraud_Label:** (0 = Not Fraudulent, 1 = Fraudulent)  

---

## Observations  
- The dataset is **highly imbalanced**, with fewer fraudulent transactions.  
- Fraudulent transactions often have **higher risk scores**, **longer distances**, and **flagged IPs**.  
- The **neural network model performs better with ReLU activation and Adam optimizer**.  
- Feature importance analysis via **permutation importance** highlights key factors.  

---

## Inference  
- The model accurately detects fraudulent transactions.  
- Fraud is correlated with **Transaction Amount, Risk Score, and Location**.  
- **Further improvements** can be made with **better feature engineering** and **oversampling (SMOTE)**.  

---

## Managerial Insights  
- **Risk-based Alerts:** Implement additional verification for high-risk transactions.  
- **Device & Location Analysis:** Flag transactions from unknown devices or locations.  
- **Fraud Prevention Strategies:** Strengthen authentication for large transactions.  
- **Real-time Monitoring:** Deploy this model to detect fraud in real-time.  

---

## How to Run the Project  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-repo/fraud-detection.git
   cd fraud-detection

