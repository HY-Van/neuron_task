"""
Classification models for neuron signal analysis
Implements Simple RNN, LSTM, and GRU models using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleRNNClassifier(nn.Module):
    """
    Simple RNN model for neuron signal classification
    """
    
    def __init__(self, 
                 input_size: int = 147,  # Number of neurons
                 hidden_size: int = 64,  # 减小隐藏层大小
                 num_layers: int = 2,
                 num_classes: int = 25,
                 dropout: float = 0.4):  # 增加dropout
        """
        Initialize Simple RNN classifier
        
        Args:
            input_size: Number of input features (neurons)
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SimpleRNNClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        
        # Take the last output
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.classifier(out)
        
        return out


class LSTMClassifier(nn.Module):
    """
    LSTM model for neuron signal classification
    """
    
    def __init__(self, 
                 input_size: int = 147,  # Number of neurons
                 hidden_size: int = 64,  # 减小隐藏层大小
                 num_layers: int = 2,
                 num_classes: int = 25,
                 dropout: float = 0.4,  # 增加dropout
                 bidirectional: bool = False):
        """
        Initialize LSTM classifier
        
        Args:
            input_size: Number of input features (neurons)
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Adjust classifier input size for bidirectional LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.classifier(out)
        
        return out


class GRUClassifier(nn.Module):
    """
    GRU model for neuron signal classification
    """
    
    def __init__(self, 
                 input_size: int = 147,  # Number of neurons
                 hidden_size: int = 64,  # 减小隐藏层大小
                 num_layers: int = 2,
                 num_classes: int = 25,
                 dropout: float = 0.4,  # 增加dropout
                 bidirectional: bool = False):
        """
        Initialize GRU classifier
        
        Args:
            input_size: Number of input features (neurons)
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Adjust classifier input size for bidirectional GRU
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the last output
        out = out[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.classifier(out)
        
        return out


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model ('rnn', 'lstm', 'gru')
        **kwargs: Model parameters
        
    Returns:
        Initialized model
    """
    models = {
        'rnn': SimpleRNNClassifier,
        'lstm': LSTMClassifier,
        'gru': GRUClassifier
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test data
    batch_size, seq_len, input_size = 32, 100, 147
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test all models
    models = ['rnn', 'lstm', 'gru']
    
    for model_name in models:
        print(f"\n{model_name.upper()} Model:")
        model = get_model(model_name)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
        
    print("\n✅ All models created successfully!")
