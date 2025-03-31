Dashboard link - https://frauddetectionmodeldeeplearning-kxs2zxcx5jb3slmfllqgzg.streamlit.app/

# Fraud Detection Model 

## **Contributors**
- Amrit Agarwal (055004)
- Oishik Banerjee (055028)

**Group No:** 23

## Project Details  
- This project focuses on building a **fraud detection model** using a **neural network** implemented with **PyTorch** and displaying performance metrics on a        streamlit dashboard. 
- The primary goal is to classify financial transactions as **fraudulent or non-fraudulent** based on transaction-related features using and artificial neural        network model and tweaking the hyperparameters to understand their effect on model accuracy.  It also helps us understand which variables form a transaction 
  record are more important in determining fraud.

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
- **ChatGPt** - Code generation. 

---


## Nature of Data -  **Imbalanced Data:** Fraudulent transactions are rare compared to legitimate ones, making detection challenging. 
### Shape and Size - (50001,21)
### Source - Kaggle
### Frequency of data - Data with daily frequency, last Updated - 31st December 2023

## Varible information
###Index - Transaction_ID, User_ID
###Non index categorical (Nominal) - Transaction_Type, Location, Device_Type, Merchant_Category, Card_Type, Authentication_Method, Is_Weekend, Fraud_Label
###Non index categorical (Ordinal) - IP_Address_Flag, Previous_Fraudulent_Activity, Daily_Transaction_Count,
###Non index non categorical - Transaction_Amount, Timestamp, Account_Balance, Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d, Card_Age,  Risk_Score, Transaction_Distance, 

## Variable Description
 
### Categorical Variables 
- **Transaction_Type** – Online, POS, ATM, etc.  
- **Device_Type** – Mobile, laptop, desktop.  
- **Location** – Geographical location of the transaction.  
- **Merchant_Category** – Merchant type.  
- **IP_Address_Flag** – Indicates if the IP is flagged.  
- **Previous_Fraudulent_Activity** – Customer’s past fraud history.  
- **Card_Type** – Credit or debit card type.  
- **Authentication_Method** – OTP, biometric, PIN.  
- **Is_Weekend** – Whether the transaction occurred on a weekend.  

### Numerical Variables 
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

## Definition of Fraud (types of fraud)
- **Failed_Transaction_Count_7d** - High Failed transactions in the last 7 days
- **Daily_Transaction_Count**  - High daily transaction count
- **Account_Balance** - Very high accounnt balance tentds to be fraud
- **Card_Type** - Mastercard has the highest
- **Risk_score**. - High risk score 

## Problem Statements  
1. **Fraud Detection:** To detect the chance of a fraud given transaction details.
2. **Classification Of Fraud**  
5. **Feature Significance Understanding:** Identifying key factors contributing to fraud can improve fraud prevention mechanisms which will help in classification of fraud.  
6. **Hyperparameter Tunig for model enhancement:** Understanding the effect of hyperparameter tuning like epochs, batch size, learning rate, hidden layers, optimizers etc on performance 
   of an artificial neural network model.  

## Model Information
### Initial
  - epochs - 5
  - batch size - 16
  - learning rate - 0.001
  - optimizer - SGD
  - activation fn - ReLu
  - neurons per layer - 20
  - dropout rate - 0.1
  - sampling method - stratified sampling (imbalanced data)

### Final
  - epochs - 50
  - batch size - 16
  - learning rate - 0.1
  - optimizer - Adam
  - activation fn - ReLu
  - neurons per layer - 100
  - dropout rate - 0.5
  - sampling method - stratified sampling (imbalanced data)

## Observations 
comment( The dataset is **highly imbalanced**, with fewer fraudulent transactions. )
- Fraudulent transactions often have high **Failed_Transaction_Count_7d**,**Daily_Transaction_Count**, **Account_Balance** **Card_Type** and  **Risk_score**.  types of fraudulent activities 
- The neural network model performs better with **ReLU activation** in dense layer and **Adam optimizer**.
- Overfitting and methods used to prevent
- The neural network model performs better with **higher numeber of epochs, higher learning rate, and higher number of neurons per layer**. 
- The highest performance of the model is 99%.
- Model Hyperparameters for best performance     **Epochs** - 50 , **Batch Size** - 16 , **Learning Rate** - 0.1, **Optimizer** - Adam, **Activation Function** -     Relu, **Neurons per Layer** - 100, **Dropout Rate** - 0.5, **Enabe Early Stopping** - Yes
- The model accurately detects fraudulent transactions.  
- Fraud is correlated with **Transaction Amount, Risk Score, and Location**.  
- **Further improvements** can be made with **better feature engineering** and **oversampling (SMOTE)**.  

---

## Managerial Insights  
- **Threshold of variables**
- **Frequency of fraudds in all categories** - Failed_Transaction_Count_7d has the highest amount of frauds and Risk_score has the least frequency of frauds
- **Transaction-based Alerts:** Implement additional verification for account that have higher number of failed transactions over  week.
- **Risk Score Monitoring:** continously monitor risk score to avoid fraudulent activities or immediate detection.
- **Transaction count Analysis:** Flag accounts with transactions in high frequency.
- **Card Type:** Additional verification of the card type that has the higher fraud activities.
- **Fraud Prevention Strategies:** Strengthen authentication for large transactions.  
- **Real-time Monitoring:** Deploy this model to detect fraud in real-time.  



