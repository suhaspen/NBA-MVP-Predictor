"""
Training script for NBA MVP prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model import MVPPredictor, MVPRanker
import os

class NBADataset(Dataset):
    """Dataset class for NBA player statistics."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(csv_path="nba_dataset.csv", use_share=True):
    """
    Load and prepare data for training.
    
    Args:
        csv_path: Path to the dataset CSV
        use_share: If True, use MVP share as target (regression). If False, use binary MVP (classification).
    
    Returns:
        X_train, X_val, y_train, y_val, scaler, feature_names
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Get feature columns (exclude identifier and target columns)
    exclude_cols = ["Player", "Year", "Tm", "MVP", "Share", "Rank"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove rows with missing values in features
    df = df.dropna(subset=feature_cols)
    
    # Select target
    if use_share:
        target_col = "Share"
        print("Using MVP Share as target (regression)")
    else:
        target_col = "MVP"
        print("Using MVP binary label as target (classification)")
    
    # Prepare features and labels
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Handle NaN in target
    if use_share:
        y = np.nan_to_num(y, nan=0.0)
    else:
        y = np.nan_to_num(y, nan=0.0).astype(int)
    
    # Split into train and validation sets
    # Use years before 2020 for training, 2020+ for validation
    train_mask = df["Year"] < 2020
    val_mask = df["Year"] >= 2020
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, X_val, y_train, y_val, scaler, feature_cols

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, use_share=True):
    """
    Train the MVP prediction model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        use_share: Whether using regression (share) or classification (binary)
    
    Returns:
        Training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Loss function and optimizer
    if use_share:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            if epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
                print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    return history

def plot_training_history(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("Saved training history plot to training_history.png")

def main():
    """Main training function."""
    # Configuration
    use_share = True  # Use MVP share (regression) for better ranking
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    
    # Load data
    X_train, X_val, y_train, y_val, scaler, feature_cols = load_data(use_share=use_share)
    
    # Create datasets and dataloaders
    train_dataset = NBADataset(X_train, y_train)
    val_dataset = NBADataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    if use_share:
        model = MVPRanker(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.3)
    else:
        model = MVPPredictor(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.3)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    history = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, use_share=use_share)
    
    # Plot training history
    plot_training_history(history)
    
    # Save scaler and feature names for inference
    import pickle
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print("\nTraining complete!")
    print("Saved files:")
    print("  - best_model.pth (model weights)")
    print("  - scaler.pkl (feature scaler)")
    print("  - feature_names.pkl (feature column names)")
    print("  - training_history.png (training curves)")

if __name__ == "__main__":
    main()
