"""
PyTorch neural network model for NBA MVP prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MVPPredictor(nn.Module):
    """
    Neural network model to predict NBA MVP.
    Uses a multi-layer perceptron with dropout for regularization.
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        """
        Initialize the MVP Predictor model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(MVPPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (binary classification: MVP or not)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

class MVPRanker(nn.Module):
    """
    Alternative model that predicts MVP share (0-1) instead of binary classification.
    This allows ranking players by their predicted MVP probability.
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        """
        Initialize the MVP Ranker model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(MVPRanker, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (regression: MVP share 0-1)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
