import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess dataset
def load_and_preprocess_data(file_path="synthetic_fraud_dataset.csv"):
    df = pd.read_csv(file_path)
    
    cat_columns = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 
                   'IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Card_Type',
                   'Authentication_Method', 'Is_Weekend', 'Fraud_Label']
    
    num_columns = ['Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 
                   'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
                   'Card_Age', 'Transaction_Distance', 'Risk_Score']

    oe = OrdinalEncoder()
    df_cat_encoded = pd.DataFrame(oe.fit_transform(df[cat_columns]), columns=cat_columns)

    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df[num_columns]), columns=num_columns)

    df_cleaned = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    
    return df_cleaned

# Streamlit UI
st.title("Fraud Detection Model Trainer")

df = load_and_preprocess_data()

# Sidebar for hyperparameters
st.sidebar.header("Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128])
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
activation_function = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])
hidden_neurons = st.sidebar.slider("Neurons per Layer", 10, 100, 50)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
early_stopping = st.sidebar.checkbox("Enable Early Stopping")

# Model Definition
class FraudNN(nn.Module):
    def __init__(self, input_size, hidden_neurons, activation_fn, dropout_rate):
        super(FraudNN, self).__init__()
        activation_dict = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid()}
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.activation = activation_dict[activation_fn]
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_neurons, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

if st.sidebar.button("Train Model"):
    try:
        X = df.drop(columns=['Fraud_Label'])
        y = df['Fraud_Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55004)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
        
        model = FraudNN(X_train.shape[1], hidden_neurons, activation_function, dropout_rate)
        optimizer_dict = {"Adam": optim.Adam, "SGD": optim.SGD, "RMSprop": optim.RMSprop}
        optimizer = optimizer_dict[optimizer_choice](model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        train_losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if early_stopping and epoch > 5 and np.mean(train_losses[-5:]) > np.mean(train_losses[-10:-5]):
                break
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
        y_pred_labels = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_labels)
        st.write(f"### Model Accuracy: {accuracy:.4f}")
        
        # Compute Permutation Feature Importance
        def compute_feature_importance(model, X, y):
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                def fit(self, X, y):
                    pass  # No need to fit again
                def predict(self, X_numpy):
                    X_tensor = torch.FloatTensor(X_numpy)
                    self.model.eval()
                    with torch.no_grad():
                        return self.model(X_tensor).numpy().flatten()
            
            wrapped_model = ModelWrapper(model)
            result = permutation_importance(wrapped_model, X, y, scoring='accuracy', n_repeats=10, random_state=42)
            return pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean}).sort_values(by="Importance", ascending=False)
        
        feature_importance = compute_feature_importance(model, X_test, y_test)
        
        st.write("### Feature Importance")
        st.dataframe(feature_importance)
        
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
        ax.set_title("Feature Importance based on Permutation Importance")
        st.pyplot(fig)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during training: {e}")
