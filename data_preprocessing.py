"""
Data preprocessing module for neuron signal analysis
Based on data exploration analysis and README requirements
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Any, Optional
import warnings
import json
import os
import pickle
import argparse
from utils import load_data, check_data_quality
warnings.filterwarnings('ignore')


class NeuronDataPreprocessor:
    """
    Simplified data preprocessor for neuron signal tasks
    Based on data exploration findings:
    - Window sizes: [24, 40, 63] based on activation duration analysis
    - No complex feature engineering
    - Handle class imbalance
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 window_size: int = 200,  # Increased for better temporal context
                 step_size: int = 1,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
                 use_first_derivative: bool = False,
                 use_second_derivative: bool = False):
        """
        Initialize the preprocessor
        
        Args:
            scaler_type: Type of scaling ('standard', 'minmax', 'none')
            window_size: Size of the sliding window (40=median, 24=25%, 63=75% from analysis)
            step_size: Step size for sliding window
            split_ratios: Ratios for train/val/test split
            use_first_derivative: Whether to include first derivative features
            use_second_derivative: Whether to include second derivative features
        """
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.step_size = step_size
        self.split_ratios = split_ratios
        self.use_first_derivative = use_first_derivative
        self.use_second_derivative = use_second_derivative
        
        # Initialize scalers
        if scaler_type == 'standard':
            self.signal_scaler = StandardScaler()
            self.position_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.signal_scaler = MinMaxScaler()
            self.position_scaler = MinMaxScaler()
        else:
            self.signal_scaler = None
            self.position_scaler = None
            
        self.label_encoder = LabelEncoder()
        
        # Store data info
        self.data_info = {}
        self.class_weights = None
        
        # Track feature dimensions
        self.original_n_features = None
        self.enhanced_n_features = None
        
    def load_and_transpose_data(self, data_path: str = 'data') -> Dict[str, np.ndarray]:
        """
        Load data and transpose signals to (time_steps, neurons) format
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Dictionary containing all loaded and transposed data
        """
        print("Loading and preprocessing data...")
        
        # Load raw data
        data_dict = load_data()
        
        # Transpose signals to (time_steps, neurons) format
        # Original: (neurons, time_steps) -> Target: (time_steps, neurons)
        data_dict['classification_signals'] = data_dict['classification_signals'].T
        data_dict['regression_signals'] = data_dict['regression_signals'].T
        
        # Store basic info
        self.data_info = {
            'n_neurons': data_dict['classification_signals'].shape[1],
            'sequence_length': data_dict['classification_signals'].shape[0],
            'n_classes': len(np.unique(data_dict['classification_labels'])),
            'position_dims': data_dict['regression_positions'].shape[1]
        }
        
        # Store original feature count
        self.original_n_features = self.data_info['n_neurons']
        
        print(f"✅ Data loaded and transposed successfully!")
        print(f"   - Sequence length: {self.data_info['sequence_length']}")
        print(f"   - Number of neurons: {self.data_info['n_neurons']}")
        print(f"   - Number of classes: {self.data_info['n_classes']}")
        print(f"   - Position dimensions: {self.data_info['position_dims']}")
        
        return data_dict
    
    def clean_data(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Simple data cleaning based on quality analysis
        
        Args:
            data_dict: Dictionary containing data arrays
            
        Returns:
            Cleaned data dictionary
        """
        print("\nPerforming data quality check and cleaning...")
        
        # First check data quality
        quality_report = check_data_quality(data_dict)
        
        cleaned_data = {}
        for key, data in data_dict.items():
            cleaned = data.copy()
            
            # Handle extreme outliers (beyond 5 standard deviations) for signals only
            if 'signals' in key:
                # Cap outliers at 5 standard deviations
                mean_val = cleaned.mean()
                std_val = cleaned.std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                
                outlier_count = ((cleaned < lower_bound) | (cleaned > upper_bound)).sum()
                if outlier_count > 0:
                    cleaned = np.clip(cleaned, lower_bound, upper_bound)
                    print(f"   - Capped {outlier_count} extreme outliers in {key}")
                else:
                    print(f"   - No extreme outliers found in {key}")
            else:
                print(f"   - {key}: No cleaning needed")
            
            cleaned_data[key] = cleaned
            
        print("✅ Data cleaning completed!")
        return cleaned_data
    
    def compute_time_domain_features(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute time-domain features for neuron signals
        
        Args:
            signals: Signal data of shape (time_steps, neurons)
            
        Returns:
            Enhanced signals with derivative features if enabled
        """
        if not self.use_first_derivative and not self.use_second_derivative:
            return signals
        
        print(f"\nComputing time-domain features...")
        print(f"   - Original signals: {signals.shape}")
        print(f"   - First derivative: {'✓' if self.use_first_derivative else '✗'}")
        print(f"   - Second derivative: {'✓' if self.use_second_derivative else '✗'}")
        
        feature_list = [signals]  # Start with original signals
        
        if self.use_first_derivative:
            # Compute first derivative using np.diff
            first_deriv = np.diff(signals, axis=0)
            # Pad with the first row to maintain the same length
            first_deriv = np.vstack([first_deriv[0:1], first_deriv])
            feature_list.append(first_deriv)
            print(f"   - First derivative computed: {first_deriv.shape}")
        
        if self.use_second_derivative:
            # Compute second derivative
            second_deriv = np.diff(signals, n=2, axis=0)
            # Pad with the first two rows to maintain the same length
            second_deriv = np.vstack([second_deriv[0:1], second_deriv[0:1], second_deriv])
            feature_list.append(second_deriv)
            print(f"   - Second derivative computed: {second_deriv.shape}")
        
        # Concatenate features along the feature dimension (last axis)
        enhanced_signals = np.concatenate(feature_list, axis=1)
        print(f"   - Enhanced signals: {enhanced_signals.shape}")
        
        return enhanced_signals
    
    def normalize_data(self, data_dict: Dict[str, np.ndarray], 
                      fit_scalers: bool = True) -> Dict[str, np.ndarray]:
        """
        Normalize/standardize the data
        
        Args:
            data_dict: Dictionary containing data arrays
            fit_scalers: Whether to fit the scalers (True for training data)
            
        Returns:
            Normalized data dictionary
        """
        if self.scaler_type == 'none':
            print("Skipping normalization...")
            return data_dict
            
        print(f"\nNormalizing data using {self.scaler_type} scaling...")
        normalized_data = {}
        
        for key, data in data_dict.items():
            if 'signals' in key:
                # Normalize signals
                if fit_scalers:
                    normalized_data[key] = self.signal_scaler.fit_transform(data)
                    print(f"   - Fitted and transformed {key}")
                else:
                    normalized_data[key] = self.signal_scaler.transform(data)
                    print(f"   - Transformed {key}")
                    
            elif 'positions' in key:
                # Normalize positions for regression
                if fit_scalers:
                    normalized_data[key] = self.position_scaler.fit_transform(data)
                    print(f"   - Fitted and transformed {key}")
                else:
                    normalized_data[key] = self.position_scaler.transform(data)
                    print(f"   - Transformed {key}")
                    
            else:
                # Labels don't need normalization
                normalized_data[key] = data
                
        print("✅ Data normalization completed!")
        return normalized_data
    
    def encode_labels_and_compute_weights(self, labels: np.ndarray, 
                                        fit_encoder: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode classification labels and compute class weights for imbalanced data
        
        Args:
            labels: Label array
            fit_encoder: Whether to fit the encoder
            
        Returns:
            Tuple of (encoded_labels, class_weights)
        """
        if fit_encoder:
            encoded = self.label_encoder.fit_transform(labels)
            
            # Compute class weights to handle imbalance (found in data exploration)
            classes = np.unique(encoded)
            class_weights = compute_class_weight('balanced', classes=classes, y=encoded)
            self.class_weights = dict(zip(classes, class_weights))
            
            print(f"   - Encoded labels: {len(np.unique(labels))} unique classes")
            print(f"   - Computed balanced class weights for imbalanced data")
            
            # Show class distribution
            unique_encoded, counts = np.unique(encoded, return_counts=True)
            min_count, max_count = counts.min(), counts.max()
            imbalance_ratio = max_count / min_count
            print(f"   - Class imbalance ratio: {imbalance_ratio:.2f}")
            
        else:
            encoded = self.label_encoder.transform(labels)
            
        return encoded, self.class_weights
    
    def create_sliding_windows(self, signals: np.ndarray, targets: np.ndarray, 
                             task_type: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window samples from time series data
        Window size based on neuron activation duration analysis from data exploration
        
        Args:
            signals: Signal data (time_steps, neurons)
            targets: Target data
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (windowed_signals, windowed_targets)
        """
        print(f"\nCreating sliding windows for {task_type}...")
        print(f"   - Window size: {self.window_size} (based on activation duration analysis)")
        print(f"   - Step size: {self.step_size}")
        
        n_samples = (len(signals) - self.window_size) // self.step_size + 1
        n_neurons = signals.shape[1]
        
        # Create windowed signals
        windowed_signals = np.zeros((n_samples, self.window_size, n_neurons), dtype=np.float32)
        
        if task_type == 'classification':
            windowed_targets = np.zeros(n_samples, dtype=np.int64)
        else:
            windowed_targets = np.zeros((n_samples, targets.shape[1]), dtype=np.float32)
        
        for i in range(n_samples):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            windowed_signals[i] = signals[start_idx:end_idx].astype(np.float32)
            windowed_targets[i] = targets[end_idx - 1]  # Target corresponds to last time step
            
        print(f"   - Generated {n_samples} samples")
        print(f"   - Sample shape: {windowed_signals.shape}")
        print(f"   - Target shape: {windowed_targets.shape}")
        
        return windowed_signals, windowed_targets
    
    def split_data_temporal(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data using temporal splitting (as required by README and data exploration)
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\nSplitting data temporally (ratios: {self.split_ratios})...")
        
        # Temporal split: train on early data, test on later data
        n_samples = len(X)
        train_end = int(n_samples * self.split_ratios[0])
        val_end = int(n_samples * (self.split_ratios[0] + self.split_ratios[1]))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        print("   - Used temporal splitting (early->train, later->test)")
        print(f"   - Train: {len(X_train)} samples")
        print(f"   - Validation: {len(X_val)} samples")
        print(f"   - Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def process_classification_data(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for classification task
        
        Args:
            data_dict: Dictionary containing raw data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print("=" * 60)
        print("PROCESSING CLASSIFICATION DATA")
        print("=" * 60)
        
        # Extract classification data
        signals = data_dict['classification_signals']
        labels = data_dict['classification_labels']
        
        # Clean data
        cleaned_data = self.clean_data({'signals': signals, 'labels': labels})
        signals = cleaned_data['signals']
        labels = cleaned_data['labels']
        
        # Apply time-domain feature engineering
        signals = self.compute_time_domain_features(signals)
        self.enhanced_n_features = signals.shape[1]
        
        # Normalize signals
        normalized_data = self.normalize_data({'signals': signals, 'labels': labels}, fit_scalers=True)
        signals = normalized_data['signals']
        
        # Encode labels and compute class weights
        encoded_labels, class_weights = self.encode_labels_and_compute_weights(labels, fit_encoder=True)
        
        # Create sliding windows
        X, y = self.create_sliding_windows(signals, encoded_labels, 'classification')
        
        # Split data temporally
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(X, y)
        
        # Prepare result dictionary
        result = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'n_classes': len(np.unique(labels)),
            'n_features': X.shape[2],  # number of enhanced features
            'original_n_features': self.original_n_features,  # original neuron count
            'enhanced_n_features': self.enhanced_n_features,  # enhanced feature count
            'window_size': self.window_size,
            'step_size': self.step_size,
            'use_first_derivative': self.use_first_derivative,
            'use_second_derivative': self.use_second_derivative,
            'class_names': self.label_encoder.classes_,
            'class_weights': class_weights,
            'data_info': self.data_info.copy()
        }
        
        print("✅ Classification data preprocessing completed!")
        return result
    
    def process_regression_data(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for regression task
        
        Args:
            data_dict: Dictionary containing raw data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print("=" * 60)
        print("PROCESSING REGRESSION DATA")
        print("=" * 60)
        
        # Extract regression data (already transposed in load_and_transpose_data)
        signals = data_dict['regression_signals']  # Already in (n_samples, n_neurons) format
        positions = data_dict['regression_positions']
        
        # Clean data
        cleaned_data = self.clean_data({'signals': signals, 'positions': positions})
        signals = cleaned_data['signals']
        positions = cleaned_data['positions']
        
        # Apply time-domain feature engineering
        signals = self.compute_time_domain_features(signals)
        self.enhanced_n_features = signals.shape[1]
        
        # Normalize data
        normalized_data = self.normalize_data({'signals': signals, 'positions': positions}, fit_scalers=True)
        signals = normalized_data['signals']
        positions = normalized_data['positions']
        
        # Create sliding windows
        X, y = self.create_sliding_windows(signals, positions, 'regression')
        
        # Split data temporally
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(X, y)
        
        # Prepare result dictionary with scaler parameters
        result = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'n_outputs': positions.shape[1],
            'n_features': X.shape[2],  # number of enhanced features
            'original_n_features': self.original_n_features,  # original neuron count
            'enhanced_n_features': self.enhanced_n_features,  # enhanced feature count
            'window_size': self.window_size,
            'step_size': self.step_size,
            'use_first_derivative': self.use_first_derivative,
            'use_second_derivative': self.use_second_derivative,
            'scaler_type': self.scaler_type,
            'data_info': self.data_info.copy()
        }
        
        # Add scaler parameters for inverse transformation
        if self.position_scaler is not None:
            if hasattr(self.position_scaler, 'mean_'):  # StandardScaler
                result['position_scaler_params'] = {
                    'mean': self.position_scaler.mean_.tolist(),
                    'scale': self.position_scaler.scale_.tolist(),
                    'var': self.position_scaler.var_.tolist()
                }
            elif hasattr(self.position_scaler, 'data_min_'):  # MinMaxScaler
                result['position_scaler_params'] = {
                    'data_min': self.position_scaler.data_min_.tolist(),
                    'data_max': self.position_scaler.data_max_.tolist(),
                    'data_range': self.position_scaler.data_range_.tolist(),
                    'scale': self.position_scaler.scale_.tolist()
                }
            
            # Add raw position statistics
            original_positions = data_dict['regression_positions']
            result['raw_position_stats'] = {
                'min': original_positions.min(axis=0).tolist(),
                'max': original_positions.max(axis=0).tolist(),
                'mean': original_positions.mean(axis=0).tolist(),
                'std': original_positions.std(axis=0).tolist()
            }
        
        if self.signal_scaler is not None:
            if hasattr(self.signal_scaler, 'mean_'):  # StandardScaler
                result['signal_scaler_params'] = {
                    'mean': self.signal_scaler.mean_.tolist(),
                    'scale': self.signal_scaler.scale_.tolist(),
                    'var': self.signal_scaler.var_.tolist()
                }
            elif hasattr(self.signal_scaler, 'data_min_'):  # MinMaxScaler
                result['signal_scaler_params'] = {
                    'data_min': self.signal_scaler.data_min_.tolist(),
                    'data_max': self.signal_scaler.data_max_.tolist(),
                    'data_range': self.signal_scaler.data_range_.tolist(),
                    'scale': self.signal_scaler.scale_.tolist()
                }
        
        print("✅ Regression data preprocessing completed!")
        return result
    
    def save_processed_data(self, processed_data: Dict[str, Any], 
                           task_type: str, output_dir: str = 'processed_data') -> None:
        """
        Save processed data to files
        
        Args:
            processed_data: Dictionary containing processed data
            task_type: 'classification' or 'regression'
            output_dir: Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed {task_type} data to {output_dir}...")
        
        # Save data arrays
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            filename = f"{output_dir}/{task_type}_{key}.npy"
            np.save(filename, processed_data[key])
            print(f"   - Saved {filename}")
        
        # Save metadata
        metadata = {k: v for k, v in processed_data.items() 
                   if not isinstance(v, np.ndarray)}
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in metadata.items():
            if key == 'class_weights' and isinstance(value, dict):
                # Convert class_weights keys from numpy int to Python int
                metadata[key] = {int(k): float(v) for k, v in value.items()}
            elif hasattr(value, 'tolist'):
                metadata[key] = value.tolist()
            elif hasattr(value, 'item'):
                metadata[key] = value.item()

        metadata_file = f"{output_dir}/{task_type}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   - Saved {metadata_file}")
        
        # Save scalers for regression task
        if task_type == 'regression' and hasattr(self, 'position_scaler') and self.position_scaler is not None:
            scaler_file = f"{output_dir}/{task_type}_position_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.position_scaler, f)
            print(f"   - Saved {scaler_file}")
        
        if hasattr(self, 'signal_scaler') and self.signal_scaler is not None:
            scaler_file = f"{output_dir}/{task_type}_signal_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.signal_scaler, f)
            print(f"   - Saved {scaler_file}")
        
        print(f"✅ {task_type.capitalize()} data saved successfully!")


def parse_arguments():
    """
    Parse command line arguments for data preprocessing
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Neuron Signal Data Preprocessing Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Window and step size parameters
    parser.add_argument('--window_size', type=int, default=200,
                       help='Size of the sliding window for time series')
    parser.add_argument('--step_size', type=int, default=1,
                       help='Step size for sliding window')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    
    # Feature engineering parameters
    parser.add_argument('--use_first_derivative', action='store_true',
                       help='Include first derivative features')
    parser.add_argument('--use_second_derivative', action='store_true',
                       help='Include second derivative features')
    
    # Data processing parameters
    parser.add_argument('--scaler_type', type=str, default='standard',
                       choices=['standard', 'minmax', 'none'],
                       help='Type of data scaling')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data')
    
    # Additional options
    parser.add_argument('--create_multiple_windows', action='store_true',
                       help='Create multiple window sizes (24, 40, 63) for experimentation')
    
    return parser.parse_args()


def create_output_directory_name(args):
    """
    Create descriptive output directory name based on parameters
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        String representing the output directory name
    """
    base_name = args.output_dir
    
    # Add window and step size info
    dir_name = f"{base_name}_w{args.window_size}_s{args.step_size}"
    
    # Add feature engineering info
    if args.use_first_derivative:
        dir_name += "_d1"
    if args.use_second_derivative:
        dir_name += "_d2"
    
    # Add scaler info if not standard
    if args.scaler_type != 'standard':
        dir_name += f"_{args.scaler_type}"
    
    return dir_name


def create_multiple_window_sizes_with_features(args):
    """
    Create preprocessed data with multiple window sizes and feature engineering
    Window sizes: [24, 40, 63] representing 25%, 50%, 75% of activation durations
    
    Args:
        args: Command line arguments containing feature engineering settings
    """
    print("=" * 80)
    print("CREATING MULTIPLE WINDOW SIZES WITH FEATURE ENGINEERING")
    print("=" * 80)
    
    # Window sizes from data exploration analysis
    window_sizes = [24, 40, 63]  # 25%, median, 75% percentiles
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    for window_size in window_sizes:
        print(f"\n{'='*20} WINDOW SIZE {window_size} {'='*20}")
        
        # Create output directory for this window size
        window_output_dir = f"{args.output_dir}_w{window_size}"
        if args.use_first_derivative:
            window_output_dir += "_d1"
        if args.use_second_derivative:
            window_output_dir += "_d2"
        
        # Initialize preprocessor
        preprocessor = NeuronDataPreprocessor(
            scaler_type=args.scaler_type,
            window_size=window_size,
            step_size=args.step_size,
            split_ratios=(args.train_ratio, args.val_ratio, test_ratio),
            use_first_derivative=args.use_first_derivative,
            use_second_derivative=args.use_second_derivative
        )
        
        # Load data
        data_dict = preprocessor.load_and_transpose_data()
        
        # Process classification data
        classification_data = preprocessor.process_classification_data(data_dict)
        preprocessor.save_processed_data(classification_data, 'classification', window_output_dir)
        
        # Reset scalers for regression
        preprocessor = NeuronDataPreprocessor(
            scaler_type=args.scaler_type,
            window_size=window_size,
            step_size=args.step_size,
            split_ratios=(args.train_ratio, args.val_ratio, test_ratio),
            use_first_derivative=args.use_first_derivative,
            use_second_derivative=args.use_second_derivative
        )
        
        # Load data again
        data_dict = preprocessor.load_and_transpose_data()
        
        # Process regression data
        regression_data = preprocessor.process_regression_data(data_dict)
        preprocessor.save_processed_data(regression_data, 'regression', window_output_dir)
        
        print(f"\n✅ Window size {window_size} completed!")
        print(f"   Output: {window_output_dir}")
        print(f"   Classification: {len(classification_data['X_train'])} train samples")
        print(f"   Regression: {len(regression_data['X_train'])} train samples")
        print(f"   Enhanced features: {regression_data['enhanced_n_features']}")


def create_multiple_window_sizes():
    """
    Create preprocessed data with multiple window sizes based on activation duration analysis
    Window sizes: [24, 40, 63] representing 25%, 50%, 75% of activation durations
    (Legacy function for backward compatibility)
    """
    print("=" * 80)
    print("CREATING MULTIPLE WINDOW SIZES BASED ON ACTIVATION DURATION ANALYSIS")
    print("=" * 80)
    
    # Window sizes from data exploration analysis
    window_sizes = [24, 40, 63]  # 25%, median, 75% percentiles
    
    for window_size in window_sizes:
        print(f"\n{'='*20} WINDOW SIZE {window_size} {'='*20}")
        
        # Initialize preprocessor
        preprocessor = NeuronDataPreprocessor(
            scaler_type='standard',
            window_size=window_size,
            step_size=1,
            split_ratios=(0.7, 0.1, 0.2),
            use_first_derivative=False,
            use_second_derivative=False
        )
        
        # Load data
        data_dict = preprocessor.load_and_transpose_data()
        
        # Process classification data
        classification_data = preprocessor.process_classification_data(data_dict)
        preprocessor.save_processed_data(classification_data, f'classification_w{window_size}')
        
        # Reset scalers for regression
        preprocessor = NeuronDataPreprocessor(
            scaler_type='standard',
            window_size=window_size,
            step_size=1,
            split_ratios=(0.7, 0.1, 0.2),
            use_first_derivative=False,
            use_second_derivative=False
        )
        
        # Load data again
        data_dict = preprocessor.load_and_transpose_data()
        
        # Process regression data
        regression_data = preprocessor.process_regression_data(data_dict)
        preprocessor.save_processed_data(regression_data, f'regression_w{window_size}')
        
        print(f"\n✅ Window size {window_size} completed!")
        print(f"   Classification: {len(classification_data['X_train'])} train samples")
        print(f"   Regression: {len(regression_data['X_train'])} train samples")


def main():
    """
    Main function - create preprocessing with command line parameters
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory name
    output_dir = create_output_directory_name(args)
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    print("=" * 80)
    print("NEURON SIGNAL DATA PREPROCESSING PIPELINE")
    print("Based on data exploration analysis with configurable parameters")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Scaler type: {args.scaler_type}")
    print(f"  - Split ratios: {args.train_ratio:.1f}/{args.val_ratio:.1f}/{test_ratio:.1f}")
    print(f"  - First derivative: {'✓' if args.use_first_derivative else '✗'}")
    print(f"  - Second derivative: {'✓' if args.use_second_derivative else '✗'}")
    print("=" * 80)
    
    # Initialize preprocessor with command line parameters
    preprocessor = NeuronDataPreprocessor(
        scaler_type=args.scaler_type,
        window_size=args.window_size,
        step_size=args.step_size,
        split_ratios=(args.train_ratio, args.val_ratio, test_ratio),
        use_first_derivative=args.use_first_derivative,
        use_second_derivative=args.use_second_derivative
    )
    
    # Load data
    data_dict = preprocessor.load_and_transpose_data()
    
    # Process classification data
    classification_data = preprocessor.process_classification_data(data_dict)
    preprocessor.save_processed_data(classification_data, 'classification', output_dir)
    
    # Reset scalers for regression (important to avoid data leakage)
    preprocessor = NeuronDataPreprocessor(
        scaler_type=args.scaler_type,
        window_size=args.window_size,
        step_size=args.step_size,
        split_ratios=(args.train_ratio, args.val_ratio, test_ratio),
        use_first_derivative=args.use_first_derivative,
        use_second_derivative=args.use_second_derivative
    )
    
    # Load data again (needed after reset)
    data_dict = preprocessor.load_and_transpose_data()
    
    # Process regression data  
    regression_data = preprocessor.process_regression_data(data_dict)
    preprocessor.save_processed_data(regression_data, 'regression', output_dir)
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"Output directory: {output_dir}")
    print(f"Classification data:")
    print(f"  - Training samples: {len(classification_data['X_train'])}")
    print(f"  - Feature dimensions: {classification_data['X_train'].shape[1:]}")
    print(f"  - Original neurons: {classification_data['original_n_features']}")
    print(f"  - Enhanced features: {classification_data['enhanced_n_features']}")
    print(f"  - Number of classes: {classification_data['n_classes']}")
    print(f"  - Class weights computed for imbalance handling")
    
    print(f"\nRegression data:")
    print(f"  - Training samples: {len(regression_data['X_train'])}")
    print(f"  - Feature dimensions: {regression_data['X_train'].shape[1:]}")
    print(f"  - Original neurons: {regression_data['original_n_features']}")
    print(f"  - Enhanced features: {regression_data['enhanced_n_features']}")
    print(f"  - Output dimensions: {regression_data['n_outputs']}")
    
    print(f"\nConfiguration used:")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Feature engineering: First deriv={'✓' if args.use_first_derivative else '✗'}, Second deriv={'✓' if args.use_second_derivative else '✗'}")
    print("Ready for model training!")
    
    # Create multiple window sizes if requested
    if args.create_multiple_windows:
        print("\n" + "="*80)
        print("CREATING MULTIPLE WINDOW SIZES FOR EXPERIMENTATION")
        print("="*80)
        create_multiple_window_sizes_with_features(args)


if __name__ == "__main__":
    # Run main preprocessing
    main()
    
    # Optionally create multiple window sizes for experimentation
    # create_multiple_window_sizes()
