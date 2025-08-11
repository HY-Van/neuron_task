"""
Training script for regression models
Trains Simple RNN, LSTM, and GRU models on neuron signal data for position prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import pickle
import warnings
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.regression_model import get_regression_model, count_parameters
from utils import load_data

warnings.filterwarnings('ignore')


def load_position_scaler(data_dir: str = 'processed_data'):
    """Load the position scaler for inverse transformation"""
    scaler_path = f"{data_dir}/regression_position_scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def inverse_transform_positions(positions_normalized: np.ndarray, scaler) -> np.ndarray:
    """Transform normalized positions back to original scale"""
    if scaler is not None:
        return scaler.inverse_transform(positions_normalized)
    else:
        return positions_normalized


class NeuronRegressionDataset(Dataset):
    """
    Dataset class for neuron signal regression data
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset
        
        Args:
            X: Input features of shape (n_samples, sequence_length, n_features)
            y: Target positions of shape (n_samples, 2) for (x, y) coordinates
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_processed_regression_data() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load preprocessed regression data
    
    Returns:
        Tuple of (data_dict, metadata)
    """
    data_dir = 'processed_data'
    
    # Load data arrays
    data_dict = {}
    for split in ['train', 'val', 'test']:
        data_dict[f'X_{split}'] = np.load(f'{data_dir}/regression_X_{split}.npy')
        data_dict[f'y_{split}'] = np.load(f'{data_dir}/regression_y_{split}.npy')
    
    # Load metadata
    with open(f'{data_dir}/regression_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return data_dict, metadata


def create_regression_data_loaders(data_dict: Dict[str, np.ndarray], 
                                 batch_size: int = 64,
                                 num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for regression
    
    Args:
        data_dict: Dictionary containing train/val/test data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = NeuronRegressionDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = NeuronRegressionDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = NeuronRegressionDataset(data_dict['X_test'], data_dict['y_test'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch_regression(model: nn.Module, 
                             train_loader: DataLoader, 
                             criterion: nn.Module, 
                             optimizer: optim.Optimizer,
                             device: torch.device) -> float:
    """
    Train regression model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}", end='\r')
        
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 添加噪声进行数据增强 (减少过拟合)
        if model.training:
            noise = torch.randn_like(X_batch) * 0.01
            X_batch = X_batch + noise
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_regression_model(model: nn.Module, 
                            val_loader: DataLoader, 
                            criterion: nn.Module,
                            device: torch.device,
                            show_original_scale: bool = True) -> Tuple[float, float, float, float, float]:
    """
    Validate regression model with both normalized and original scale metrics
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        show_original_scale: Whether to compute original scale metrics
        
    Returns:
        Tuple of (average_loss, mae_norm, mse_norm, mae_orig, mse_orig)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            
            # Collect predictions and targets for metrics
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Calculate normalized metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mae_norm = mean_absolute_error(targets, predictions)
    mse_norm = mean_squared_error(targets, predictions)
    avg_loss = total_loss / len(val_loader)
    
    # Calculate original scale metrics if requested
    mae_orig, mse_orig = 0.0, 0.0
    if show_original_scale:
        try:
            # Load position scaler to inverse transform
            position_scaler = load_position_scaler()
            if position_scaler is not None:
                targets_orig = position_scaler.inverse_transform(targets)
                predictions_orig = position_scaler.inverse_transform(predictions)
                mae_orig = mean_absolute_error(targets_orig, predictions_orig)
                mse_orig = mean_squared_error(targets_orig, predictions_orig)
        except Exception as e:
            print(f"Warning: Could not compute original scale metrics: {e}")
    
    return avg_loss, mae_norm, mse_norm, mae_orig, mse_orig


def train_regression_model(model_name: str,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         device: torch.device,
                         num_epochs: int = 50,
                         learning_rate: float = 0.0001,
                         patience: int = 15,
                         weight_decay: float = 1e-4) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a regression model with early stopping
    
    Args:
        model_name: Name of the model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        weight_decay: L2 regularization weight decay
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create model
    model = get_regression_model(model_name)
    model.to(device)
    
    print(f"\n🔧 Training {model_name.upper()} Regression Model")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Loss function for regression (MSE)
    criterion = nn.MSELoss()
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler - more conservative
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mse': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train_one_epoch_regression(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae_norm, val_mse_norm, val_mae_orig, val_mse_orig = validate_regression_model(
            model, val_loader, criterion, device, show_original_scale=True)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae_norm)
        history['val_mse'].append(val_mse_norm)
        
        # Print progress with overfitting detection
        overfitting_warning = ""
        if epoch > 0 and val_loss > history['val_loss'][-2] and train_loss < history['train_loss'][-2]:
            overfitting_warning = " 🚨 Overfitting detected!"
        
        # Prepare metrics display
        metrics_display = f"MAE_norm: {val_mae_norm:.4f}, MSE_norm: {val_mse_norm:.4f}"
        if val_mae_orig > 0 and val_mse_orig > 0:
            metrics_display += f", MAE_orig: {val_mae_orig:.2f}, MSE_orig: {val_mse_orig:.2f}"
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"{metrics_display}{overfitting_warning}")
        
        # Early stopping check with overfitting prevention
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        # 如果连续3个epoch都在过拟合，降低学习率
        if epoch >= 2:
            recent_train_losses = history['train_loss'][-3:]
            recent_val_losses = history['val_loss'][-3:]
            if all(t1 < t2 for t1, t2 in zip(recent_train_losses[1:], recent_train_losses[:-1])) and \
               all(v1 > v2 for v1, v2 in zip(recent_val_losses[1:], recent_val_losses[:-1])):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"   Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
            
        if patience_counter >= patience:
            print(f"⏰ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ {model_name.upper()} regression training completed!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    
    return model, history


def save_regression_model_and_history(model: nn.Module, 
                                    history: Dict[str, Any], 
                                    model_name: str,
                                    save_dir: str = 'trained_models') -> None:
    """
    Save trained regression model and training history
    
    Args:
        model: Trained model
        history: Training history
        model_name: Name of the model
        save_dir: Directory to save to
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = f"{save_dir}/{model_name}_regressor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   Model saved to: {model_path}")
    
    # Save history
    history_path = f"{save_dir}/{model_name}_regression_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History saved to: {history_path}")


def plot_regression_training_history(histories: Dict[str, Dict[str, Any]], 
                                   save_path: str = 'regression_training_plots.png') -> None:
    """
    Plot training histories for all regression models
    
    Args:
        histories: Dictionary of model histories
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs, history['train_loss'], label=f'{model_name.upper()} Train')
        axes[0, 0].plot(epochs, history['val_loss'], label=f'{model_name.upper()} Val')
    
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot MAE
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_mae']) + 1)
        axes[0, 1].plot(epochs, history['val_mae'], label=f'{model_name.upper()}')
    
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot MSE
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_mse']) + 1)
        axes[1, 0].plot(epochs, history['val_mse'], label=f'{model_name.upper()}')
    
    axes[1, 0].set_title('Validation MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Squared Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot learning curves (train vs val loss comparison)
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        axes[1, 1].plot(epochs, val_loss - train_loss, label=f'{model_name.upper()} Gap')
    
    axes[1, 1].set_title('Train-Val Loss Gap (Overfitting Indicator)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Regression training plots saved to: {save_path}")


def main():
    """
    Main regression training function
    """
    print("=" * 80)
    print("TRAINING REGRESSION MODELS")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("\n📁 Loading preprocessed regression data...")
    data_dict, metadata = load_processed_regression_data()
    
    print(f"   Training samples: {data_dict['X_train'].shape[0]:,}")
    print(f"   Validation samples: {data_dict['X_val'].shape[0]:,}")
    print(f"   Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   Sequence length: {data_dict['X_train'].shape[1]}")
    print(f"   Number of features: {data_dict['X_train'].shape[2]}")
    print(f"   Output dimensions: {data_dict['y_train'].shape[1]} (x, y coordinates)")
    
    # Print position statistics
    print(f"\n📊 Position Statistics:")
    print(f"   X range: [{data_dict['y_train'][:, 0].min():.2f}, {data_dict['y_train'][:, 0].max():.2f}]")
    print(f"   Y range: [{data_dict['y_train'][:, 1].min():.2f}, {data_dict['y_train'][:, 1].max():.2f}]")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    batch_size = 64 if device.type == 'cuda' else 32
    train_loader, val_loader, test_loader = create_regression_data_loaders(data_dict, batch_size=batch_size)
    
    # Models to train
    models_to_train = ['rnn', 'lstm', 'gru']
    trained_models = {}
    histories = {}
    
    # Train each model
    for model_name in models_to_train:
        try:
            model, history = train_regression_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=50,
                learning_rate=0.0001,
                patience=15,
                weight_decay=1e-4
            )
            
            # Save model and history
            save_regression_model_and_history(model, history, model_name)
            
            # Store for comparison
            trained_models[model_name] = model
            histories[model_name] = history
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Plot training histories
    if histories:
        print("\n📊 Creating regression training plots...")
        plot_regression_training_history(histories)
    
    # Summary
    print("\n" + "=" * 80)
    print("REGRESSION TRAINING SUMMARY")
    print("=" * 80)
    
    for model_name, history in histories.items():
        best_mae = min(history['val_mae'])
        best_mse = min(history['val_mse'])
        best_val_loss = min(history['val_loss'])
        print(f"{model_name.upper():>8}: Best MAE: {best_mae:.4f}, Best MSE: {best_mse:.4f}, Best Val Loss: {best_val_loss:.6f}")
    
    print(f"\n✅ Regression training completed! Models saved in 'trained_models/' directory")
    print("   Run 'python evaluate.py' to evaluate the models on test data")


if __name__ == "__main__":
    main()
