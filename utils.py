"""
Utility functions for neuron signal analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==============================================================================
# DATA CONFIGURATION - CENTRALIZED PATH MANAGEMENT
# ==============================================================================
DATA_PATHS = {
    'raw_data': 'data',
    'processed_data': 'data_processed/processed_data_w80_s1_d1',  # Default processed data directory
    'trained_models': 'trained_models',
    'evaluation_plots': 'evaluation_plots'
}

DATASET_CONFIG = {
    'window_size': 80,
    'n_neurons': 147,
    'n_classes': 25,
    'n_position_dims': 2,
    'split_ratios': [0.7, 0.1, 0.2]  # train, val, test
}

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_raw_data(data_dir: str = None) -> Dict[str, np.ndarray]:
    """
    Load raw data files for the neuron task
    
    Args:
        data_dir: Directory containing raw data files
        
    Returns:
        Dict containing all loaded data arrays
    """
    if data_dir is None:
        data_dir = DATA_PATHS['raw_data']
    
    data_dict = {}
    
    # Classification data
    print("Loading classification data...")
    data_dict['classification_signals'] = np.load(f'{data_dir}/classification/neuron_signals_aligned.npy')
    data_dict['classification_labels'] = np.load(f'{data_dir}/classification/grid_sequence.npy')
    
    # Regression data
    print("Loading regression data...")
    data_dict['regression_signals'] = np.load(f'{data_dir}/regression/neuron_signals_aligned.npy')
    data_dict['regression_positions'] = np.load(f'{data_dir}/regression/positions_cm.npy')
    
    return data_dict

def load_processed_data(data_dir: str = None, task_type: str = 'classification') -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load preprocessed data for training/evaluation
    
    Args:
        data_dir: Directory containing processed data
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (data_dict, metadata)
    """
    if data_dir is None:
        data_dir = DATA_PATHS['processed_data']
    
    # Load data arrays
    data_dict = {}
    for split in ['train', 'val', 'test']:
        data_dict[f'X_{split}'] = np.load(f'{data_dir}/{task_type}_X_{split}.npy')
        data_dict[f'y_{split}'] = np.load(f'{data_dir}/{task_type}_y_{split}.npy')
    
    # Load metadata
    with open(f'{data_dir}/{task_type}_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return data_dict, metadata

def load_scaler(data_dir: str = None, task_type: str = 'regression', scaler_type: str = 'position'):
    """
    Load fitted scaler for data transformation
    
    Args:
        data_dir: Directory containing scaler files
        task_type: 'classification' or 'regression'
        scaler_type: 'position' or 'signal'
        
    Returns:
        Fitted scaler object or None if not found
    """
    if data_dir is None:
        data_dir = DATA_PATHS['processed_data']
    
    scaler_path = f"{data_dir}/{task_type}_{scaler_type}_scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

# ==============================================================================
# PYTORCH DATASET CLASSES
# ==============================================================================

class NeuronDataset(Dataset):
    """Universal dataset class for neuron signal data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str = 'classification'):
        """
        Args:
            X: Input features of shape (n_samples, sequence_length, n_features)
            y: Labels/targets
            task_type: 'classification' or 'regression'
        """
        self.X = torch.FloatTensor(X)
        
        if task_type == 'classification':
            self.y = torch.LongTensor(y)
        else:  # regression
            self.y = torch.FloatTensor(y)
        
        self.task_type = task_type
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(data_dict: Dict[str, np.ndarray], 
                       task_type: str = 'classification',
                       batch_size: int = 64,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders
    
    Args:
        data_dict: Dictionary containing train/val/test data
        task_type: 'classification' or 'regression'
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = NeuronDataset(data_dict['X_train'], data_dict['y_train'], task_type)
    val_dataset = NeuronDataset(data_dict['X_val'], data_dict['y_val'], task_type)
    test_dataset = NeuronDataset(data_dict['X_test'], data_dict['y_test'], task_type)
    
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

# ==============================================================================
# EVALUATION UTILITIES
# ==============================================================================

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler=None) -> Dict[str, float]:
    """Calculate comprehensive regression metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Original scale metrics (priority)
    if scaler is not None:
        y_true_orig = scaler.inverse_transform(y_true)
        y_pred_orig = scaler.inverse_transform(y_pred)
        
        # Calculate Euclidean distance for position prediction
        distances = np.sqrt(np.sum((y_pred_orig - y_true_orig)**2, axis=1))
        mae_orig = np.mean(distances)
        mse_orig = np.mean(np.sum((y_pred_orig - y_true_orig)**2, axis=1))
        rmse_orig = np.sqrt(mse_orig)
        r2_orig = r2_score(y_true_orig, y_pred_orig)
    else:
        mae_orig = mse_orig = rmse_orig = r2_orig = None
    
    # Normalized metrics
    mae_norm = mean_absolute_error(y_true, y_pred)
    mse_norm = mean_squared_error(y_true, y_pred)
    rmse_norm = np.sqrt(mse_norm)
    r2_norm = r2_score(y_true, y_pred)
    
    metrics = {
        'mae_normalized': mae_norm,
        'mse_normalized': mse_norm,
        'rmse_normalized': rmse_norm,
        'r2_normalized': r2_norm
    }
    
    if scaler is not None:
        metrics.update({
            'mae_original': mae_orig,
            'mse_original': mse_orig,
            'rmse_original': rmse_orig,
            'r2_original': r2_orig
        })
    
    return metrics

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def save_training_plots(history: Dict[str, List], model_name: str, task_type: str):
    """Save training history plots"""
    os.makedirs(DATA_PATHS['evaluation_plots'], exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name.upper()} - Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metric plot
    if task_type == 'classification' and 'val_accuracy' in history:
        axes[1].plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='green')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{model_name.upper()} - Validation Accuracy')
    elif task_type == 'regression' and 'val_mae' in history:
        axes[1].plot(epochs, history['val_mae'], label='Validation MAE', color='orange')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{model_name.upper()} - Validation MAE')
    
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    save_path = f"{DATA_PATHS['evaluation_plots']}/{model_name}_{task_type}_training.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved: {save_path}")

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_and_results(model: nn.Module, 
                          history: Dict[str, List], 
                          model_name: str,
                          task_type: str,
                          metadata: Dict[str, Any] = None):
    """Save trained model and training history"""
    os.makedirs(DATA_PATHS['trained_models'], exist_ok=True)
    
    # Save model
    model_path = f"{DATA_PATHS['trained_models']}/{model_name}_{task_type}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save history
    history_path = f"{DATA_PATHS['trained_models']}/{model_name}_{task_type}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📋 History saved: {history_path}")
    
    # Save training plots
    save_training_plots(history, model_name, task_type)

def print_data_info(data_dict: Dict[str, np.ndarray]) -> None:
    """Print basic information about loaded data"""
    print("=" * 60)
    print("DATA INFORMATION SUMMARY")
    print("=" * 60)
    
    for key, data in data_dict.items():
        print(f"\n{key.upper()}:")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Min value: {data.min():.4f}")
        print(f"  Max value: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
        
        # Check for NaN values
        nan_count = np.isnan(data).sum()
        print(f"  NaN values: {nan_count}")
        
        # Memory usage
        memory_mb = data.nbytes / (1024 * 1024)
        print(f"  Memory usage: {memory_mb:.2f} MB")

# Legacy function for backward compatibility
def load_data():
    """Legacy function - use load_raw_data() instead"""
    return load_raw_data()


# ==============================================================================
# USED IN DATA_EXPLORATION.IPYNB
# ==============================================================================

def analyze_signal_statistics(signals: np.ndarray, title: str = "Signal") -> pd.DataFrame:
    """
    Analyze statistical properties of neuron signals
    
    Args:
        signals: Signal data array (time_steps, neurons)
        title: Title for the analysis
        
    Returns:
        DataFrame with statistical summary
    """
    print(f"\n{title} Statistical Analysis:")
    print("-" * 40)
    
    # Basic statistics for each neuron
    stats_df = pd.DataFrame({
        'mean': signals.mean(axis=0),
        'std': signals.std(axis=0),
        'min': signals.min(axis=0),
        'max': signals.max(axis=0),
        'median': np.median(signals, axis=0),
        'q25': np.percentile(signals, 25, axis=0),
        'q75': np.percentile(signals, 75, axis=0)
    })
    
    # Overall statistics
    print(f"Number of neurons: {signals.shape[1]}")
    print(f"Number of time steps: {signals.shape[0]}")
    print(f"Overall signal range: [{signals.min():.4f}, {signals.max():.4f}]")
    print(f"Overall mean: {signals.mean():.4f}")
    print(f"Overall std: {signals.std():.4f}")
    
    return stats_df


def plot_signal_overview(signals: np.ndarray, title: str = "Neuron Signals", 
                        max_neurons: int = 20, figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot overview of neuron signals
    
    Args:
        signals: Signal data array (time_steps, neurons)
        title: Plot title
        max_neurons: Maximum number of neurons to plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title} Overview', fontsize=16)
    
    # Plot 1: First few neurons over time
    axes[0, 0].plot(signals[:1000, :min(max_neurons, signals.shape[1])])
    axes[0, 0].set_title(f'First {min(max_neurons, signals.shape[1])} Neurons (First 1000 timesteps)')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Signal Amplitude')
    
    # Plot 2: Signal distribution histogram
    axes[0, 1].hist(signals.flatten(), bins=500, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Signal Distribution')
    axes[0, 1].set_xlabel('Signal Amplitude')
    axes[0, 1].set_xlim([0, 5])
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Mean signal per neuron
    mean_signals = signals.mean(axis=0)
    axes[1, 0].bar(range(len(mean_signals)), mean_signals)
    axes[1, 0].set_title('Mean Signal per Neuron')
    axes[1, 0].set_xlabel('Neuron Index')
    axes[1, 0].set_ylabel('Mean Signal')
    
    # Plot 4: Signal variance per neuron
    var_signals = signals.var(axis=0)
    axes[1, 1].bar(range(len(var_signals)), var_signals)
    axes[1, 1].set_title('Signal Variance per Neuron')
    axes[1, 1].set_xlabel('Neuron Index')
    axes[1, 1].set_ylabel('Signal Variance')
    
    plt.tight_layout()
    plt.show()


def analyze_correlation_matrix(signals: np.ndarray, title: str = "Neuron Signals", 
                              sample_size: int = 5000) -> np.ndarray:
    """
    Analyze correlation between neurons
    
    Args:
        signals: Signal data array (time_steps, neurons)
        title: Title for the analysis
        sample_size: Number of samples to use for correlation analysis
        
    Returns:
        Correlation matrix
    """
    # Sample data if too large
    if signals.shape[0] > sample_size:
        indices = np.random.choice(signals.shape[0], sample_size, replace=False)
        sample_signals = signals[indices]
    else:
        sample_signals = signals
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(sample_signals.T)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, cbar_kws={"shrink": .8})
    plt.title(f'{title} - Neuron Correlation Matrix')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.show()
    
    # Print correlation statistics
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    print(f"\nCorrelation Statistics for {title}:")
    print(f"Mean correlation: {upper_triangle.mean():.4f}")
    print(f"Std correlation: {upper_triangle.std():.4f}")
    print(f"Max correlation: {upper_triangle.max():.4f}")
    print(f"Min correlation: {upper_triangle.min():.4f}")
    
    return corr_matrix


def plot_target_distribution(targets: np.ndarray, title: str = "Target Distribution",
                           is_categorical: bool = False) -> None:
    """
    Plot distribution of target variables
    
    Args:
        targets: Target data array
        title: Plot title
        is_categorical: Whether targets are categorical (classification) or continuous (regression)
    """
    if is_categorical:
        # For classification targets
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        unique, counts = np.unique(targets, return_counts=True)
        plt.bar(unique, counts)
        plt.title(f'{title} - Class Distribution')
        plt.xlabel('Grid ID')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=unique, autopct='%1.1f%%')
        plt.title(f'{title} - Class Proportion')
        
    else:
        # For regression targets (2D positions)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # X coordinate distribution
        axes[0, 0].hist(targets[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('X Coordinate Distribution')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Y coordinate distribution
        axes[0, 1].hist(targets[:, 1], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Y Coordinate Distribution')
        axes[0, 1].set_xlabel('Y Position (m)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 2D scatter plot of positions
        axes[1, 0].scatter(targets[:, 0], targets[:, 1], alpha=0.5, s=1)
        axes[1, 0].set_title('2D Position Distribution')
        axes[1, 0].set_xlabel('X Position (m)')
        axes[1, 0].set_ylabel('Y Position (m)')
        
        # Position trajectory (first 1000 points)
        n_points = min(1000, len(targets))
        axes[1, 1].plot(targets[:n_points, 0], targets[:n_points, 1], '-', alpha=0.7, linewidth=0.5)
        axes[1, 1].scatter(targets[0, 0], targets[0, 1], color='green', s=50, label='Start')
        axes[1, 1].scatter(targets[n_points-1, 0], targets[n_points-1, 1], color='red', s=50, label='End')
        axes[1, 1].set_title(f'Position Trajectory (First {n_points} points)')
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def check_data_quality(data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Comprehensive data quality check
    
    Args:
        data_dict: Dictionary containing all data arrays
        
    Returns:
        Dictionary with quality check results
    """
    quality_report = {}
    
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    for key, data in data_dict.items():
        print(f"\nAnalyzing {key}...")
        
        quality_info = {
            'shape': data.shape,
            'has_nan': np.isnan(data).any(),
            'nan_count': np.isnan(data).sum(),
            'has_inf': np.isinf(data).any(),
            'inf_count': np.isinf(data).sum(),
            'unique_values': len(np.unique(data)) if data.size < 1000000 else "Too large to compute",
            'zero_values': (data == 0).sum(),
            'negative_values': (data < 0).sum() if not key.endswith('labels') else "N/A"
        }
        
        quality_report[key] = quality_info
        
        # Print quality issues
        if quality_info['has_nan']:
            print(f"  ⚠️  Found {quality_info['nan_count']} NaN values")
        if quality_info['has_inf']:
            print(f"  ⚠️  Found {quality_info['inf_count']} infinite values")
        if quality_info['zero_values'] > data.size * 0.1:
            print(f"  ⚠️  High number of zero values: {quality_info['zero_values']} ({100*quality_info['zero_values']/data.size:.1f}%)")
        
        if not quality_info['has_nan'] and not quality_info['has_inf']:
            print(f"  ✅ Data quality looks good")
    
    return quality_report

