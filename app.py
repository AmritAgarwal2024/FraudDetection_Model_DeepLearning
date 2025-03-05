import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to load dataset from streamlit

def load_data(file_path="synthetic_fraud_dataset.csv"):
    return pd.read_csv(file_path)

# Streamlit UI
st.title("Fraud Detection Model Trainer")


df = load_data()
      
      

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

# Model Training (Executed on button click)
if st.sidebar.button("Train Model") and url:
    try:
        # Preprocess dataset
        X = df.drop(columns=['label'])  # Assuming 'label' column is target
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
        
        # Define model
        model = FraudNN(X_train.shape[1], hidden_neurons, activation_function, dropout_rate)
        optimizer_dict = {"Adam": optim.Adam, "SGD": optim.SGD, "RMSprop": optim.RMSprop}
        optimizer = optimizer_dict[optimizer_choice](model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
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
        
        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='Training Loss')
        ax.set_title("Loss Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during training: {e}")
