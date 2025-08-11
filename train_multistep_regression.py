"""
Multi-Step Regression Model Training Script
Train models to predict 30 future steps of 2D positions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import warnings
from tqdm import tqdm

# Import utils with centralized configuration
from utils import (
    DATA_PATHS, DATASET_CONFIG,
    load_processed_data, create_data_loaders, load_scaler,
    calculate_regression_metrics, save_model_and_results,
    count_parameters
)

# Import models with multi-step support
from models.LSTM import get_lstm_model
from models.GRU import get_gru_model

warnings.filterwarnings('ignore')

def create_multistep_data(data_dict, pred_len=30):
    """
    Create multi-step prediction data from single-step data
    
    Args:
        data_dict: Original data dictionary
        pred_len: Number of future steps to predict
        
    Returns:
        Modified data dictionary for multi-step prediction
    """
    print(f"Creating multi-step data with prediction length: {pred_len}")
    
    # Get original data
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    
    def create_sequences(X, y, pred_len):
        """Create sequences for multi-step prediction"""
        X_seq, y_seq = [], []
        
        # We need at least pred_len future points for each sequence
        max_idx = len(X) - pred_len + 1
        
        for i in range(max_idx):
            X_seq.append(X[i])
            # Create target sequence: next pred_len positions
            y_seq.append(y[i:i+pred_len])
        
        return np.array(X_seq), np.array(y_seq)
    
    # Create multi-step sequences
    X_train_ms, y_train_ms = create_sequences(X_train, y_train, pred_len)
    X_val_ms, y_val_ms = create_sequences(X_val, y_val, pred_len)
    X_test_ms, y_test_ms = create_sequences(X_test, y_test, pred_len)
    
    print(f"Multi-step data shapes:")
    print(f"  Train: X{X_train_ms.shape} -> y{y_train_ms.shape}")
    print(f"  Val:   X{X_val_ms.shape} -> y{y_val_ms.shape}")
    print(f"  Test:  X{X_test_ms.shape} -> y{y_test_ms.shape}")
    
    # Create new data dictionary
    multistep_data = {
        'X_train': X_train_ms,
        'X_val': X_val_ms,
        'X_test': X_test_ms,
        'y_train': y_train_ms,
        'y_val': y_val_ms,
        'y_test': y_test_ms
    }
    
    return multistep_data

def create_multistep_data_loaders(data_dict, batch_size=32):
    """Create data loaders for multi-step prediction"""
    
    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_dict['X_train']),
        torch.FloatTensor(data_dict['y_train'])
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_dict['X_val']),
        torch.FloatTensor(data_dict['y_val'])
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_dict['X_test']),
        torch.FloatTensor(data_dict['y_test'])
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_teacher_forcing=True):
    """Train model for one epoch with teacher forcing"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        if use_teacher_forcing:
            predictions = model(X_batch, target_seq=y_batch, use_teacher_forcing=True)
        else:
            predictions = model(X_batch, use_teacher_forcing=False)
        
        # Calculate loss
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def validate_model(model, val_loader, criterion, device, position_scaler=None):
    """Validate model without teacher forcing and return loss, MAE, MSE metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass without teacher forcing (real inference)
            predictions = model(X_batch, use_teacher_forcing=False)
            
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions and targets for metric calculation
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Calculate MAE and MSE metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metric calculation
    pred_flat = predictions.reshape(-1, 2)
    targ_flat = targets.reshape(-1, 2)
    
    # Calculate MAE (Mean Absolute Error for distance)
    mae_normalized = np.mean(np.sqrt(np.sum((pred_flat - targ_flat)**2, axis=1)))
    
    # Calculate MSE (Mean Squared Error)
    mse_normalized = np.mean(np.sum((pred_flat - targ_flat)**2, axis=1))
    
    # Convert to original scale if scaler available
    mae_original = None
    mse_original = None
    if position_scaler is not None:
        pred_orig = position_scaler.inverse_transform(pred_flat)
        targ_orig = position_scaler.inverse_transform(targ_flat)
        
        mae_original = np.mean(np.sqrt(np.sum((pred_orig - targ_orig)**2, axis=1)))
        mse_original = np.mean(np.sum((pred_orig - targ_orig)**2, axis=1))
    
    avg_loss = total_loss / num_batches
    return avg_loss, mae_normalized, mse_normalized, mae_original, mse_original

def evaluate_multistep_predictions(predictions, targets, position_scaler=None):
    """Evaluate multi-step predictions with comprehensive metrics"""
    # predictions and targets shape: (batch_size, pred_len, 2)
    
    # Flatten for overall evaluation
    pred_flat = predictions.reshape(-1, 2)
    targ_flat = targets.reshape(-1, 2)
    
    # Calculate overall metrics using existing function
    metrics = calculate_regression_metrics(targ_flat, pred_flat, position_scaler)
    
    # Additional multi-step specific metrics
    pred_len = predictions.shape[1]
    step_wise_mae_norm = []
    step_wise_mse_norm = []
    step_wise_mae_orig = []
    step_wise_mse_orig = []
    
    for step in range(pred_len):
        step_pred = predictions[:, step, :]  # (batch_size, 2)
        step_targ = targets[:, step, :]      # (batch_size, 2)
        
        # Normalized metrics for this step
        step_mae_norm = np.mean(np.sqrt(np.sum((step_pred - step_targ)**2, axis=1)))
        step_mse_norm = np.mean(np.sum((step_pred - step_targ)**2, axis=1))
        step_wise_mae_norm.append(step_mae_norm)
        step_wise_mse_norm.append(step_mse_norm)
        
        # Original scale metrics if scaler available
        if position_scaler is not None:
            step_pred_orig = position_scaler.inverse_transform(step_pred)
            step_targ_orig = position_scaler.inverse_transform(step_targ)
            
            step_mae_orig = np.mean(np.sqrt(np.sum((step_pred_orig - step_targ_orig)**2, axis=1)))
            step_mse_orig = np.mean(np.sum((step_pred_orig - step_targ_orig)**2, axis=1))
            step_wise_mae_orig.append(step_mae_orig)
            step_wise_mse_orig.append(step_mse_orig)
    
    # Add step-wise metrics to the main metrics dictionary
    metrics['step_wise_mae_normalized'] = step_wise_mae_norm
    metrics['step_wise_mse_normalized'] = step_wise_mse_norm
    metrics['final_step_mae_normalized'] = step_wise_mae_norm[-1]  # MAE at final step
    metrics['final_step_mse_normalized'] = step_wise_mse_norm[-1]  # MSE at final step
    
    if position_scaler is not None:
        metrics['step_wise_mae_original'] = step_wise_mae_orig
        metrics['step_wise_mse_original'] = step_wise_mse_orig
        metrics['final_step_mae_original'] = step_wise_mae_orig[-1]
        metrics['final_step_mse_original'] = step_wise_mse_orig[-1]
    
    # Calculate trajectory-level metrics
    # Average deviation per trajectory
    traj_errors = []
    for i in range(predictions.shape[0]):
        traj_pred = predictions[i]  # (pred_len, 2)
        traj_targ = targets[i]      # (pred_len, 2)
        traj_error = np.mean(np.sqrt(np.sum((traj_pred - traj_targ)**2, axis=1)))
        traj_errors.append(traj_error)
    
    metrics['trajectory_mae_normalized'] = np.mean(traj_errors)
    metrics['trajectory_mae_std'] = np.std(traj_errors)
    
    return metrics

def plot_multistep_predictions(predictions, targets, model_name, save_dir, position_scaler=None, max_samples=5):
    """Plot multi-step trajectory predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use original scale if available
    if position_scaler is not None:
        pred_orig = position_scaler.inverse_transform(predictions.reshape(-1, 2)).reshape(predictions.shape)
        targ_orig = position_scaler.inverse_transform(targets.reshape(-1, 2)).reshape(targets.shape)
        unit = "cm"
    else:
        pred_orig = predictions
        targ_orig = targets
        unit = "(normalized)"
    
    # Select random samples for visualization
    n_samples = min(max_samples, predictions.shape[0])
    sample_indices = np.random.choice(predictions.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for i, sample_idx in enumerate(sample_indices):
        pred_traj = pred_orig[sample_idx]  # (pred_len, 2)
        true_traj = targ_orig[sample_idx]  # (pred_len, 2)
        
        # Plot trajectories
        axes[i].plot(true_traj[:, 0], true_traj[:, 1], 'b-o', 
                    markersize=3, linewidth=2, label='True Trajectory')
        axes[i].plot(pred_traj[:, 0], pred_traj[:, 1], 'r-s', 
                    markersize=3, linewidth=2, label='Predicted Trajectory')
        
        # Mark start and end points
        axes[i].scatter(true_traj[0, 0], true_traj[0, 1], 
                       color='blue', s=100, marker='*', label='Start (True)', zorder=5)
        axes[i].scatter(pred_traj[0, 0], pred_traj[0, 1], 
                       color='red', s=100, marker='*', label='Start (Pred)', zorder=5)
        
        axes[i].set_xlabel(f'X Position {unit}')
        axes[i].set_ylabel(f'Y Position {unit}')
        axes[i].set_title(f'Sample {sample_idx + 1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')
    
    plt.suptitle(f'{model_name.upper()} - 30-Step Trajectory Predictions')
    plt.tight_layout()
    
    save_path = f"{save_dir}/{model_name}_multistep_trajectories.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Multi-step trajectory plots saved: {save_path}")

def train_multistep_model(model_name, pred_len=30, num_epochs=100, batch_size=32, patience=20):
    """Train a multi-step prediction model"""
    
    print(f"\n🚀 Training {model_name.upper()} for {pred_len}-step prediction")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load single-step data
    print(f"\n📁 Loading data from {DATA_PATHS['processed_data']}...")
    data_dict, metadata = load_processed_data(None, 'regression')
    
    # Create multi-step data
    multistep_data = create_multistep_data(data_dict, pred_len)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_multistep_data_loaders(
        multistep_data, batch_size
    )
    
    # Create model
    input_size = multistep_data['X_train'].shape[2]
    
    print(f"\n🔧 Creating {model_name.upper()} model...")
    if model_name == 'lstm':
        model = get_lstm_model('multistep_regression', 
                              input_size=input_size, 
                              output_size=2, 
                              pred_len=pred_len)
    elif model_name == 'gru':
        model = get_gru_model('multistep_regression', 
                             input_size=input_size, 
                             output_size=2, 
                             pred_len=pred_len)
    else:
        raise ValueError(f"Model {model_name} not supported for multi-step prediction yet")
    
    model.to(device)
    
    param_count = count_parameters(model)
    print(f"   Parameters: {param_count:,}")
    print(f"   Prediction steps: {pred_len}")
    
    # Setup training
    criterion = nn.MSELoss()
    
    # Conservative learning rate for multi-step prediction
    lr = 0.0005 if model_name == 'lstm' else 0.0003
    print(f"   Learning rate: {lr}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Scheduler for multi-step training - fixed to remove verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Load position scaler for validation metrics
    position_scaler = load_scaler(None, 'regression', 'position')
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_teacher_forcing=True  # Use teacher forcing during training
        )
        
        # Validation (without teacher forcing) - now returns MAE/MSE metrics
        val_loss, val_mae, val_mse, val_mae_orig, val_mse_orig = validate_model(
            model, val_loader, criterion, device, position_scaler
        )
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change
        if current_lr != old_lr:
            print(f"   Learning rate reduced to: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        # Print progress with MAE and MSE metrics
        progress_str = (f"Epoch {epoch+1:3d}/{num_epochs}: "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, ")
        
        if val_mae_orig is not None and val_mse_orig is not None:
            # Show original scale metrics if available
            progress_str += (f"Val MAE: {val_mae_orig:.2f}cm, "
                           f"Val MSE: {val_mse_orig:.2f}cm², ")
        else:
            # Show normalized metrics
            progress_str += (f"Val MAE: {val_mae:.4f}, "
                           f"Val MSE: {val_mse:.4f}, ")
        
        progress_str += f"LR: {current_lr:.6f}"
        print(progress_str)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print(f"\n📊 Final evaluation...")
    model.eval()
    position_scaler = load_scaler(None, 'regression', 'position')
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch, use_teacher_forcing=False)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Evaluate predictions
    metrics = evaluate_multistep_predictions(predictions, targets, position_scaler)
    
    print(f"   📈 Results:")
    if position_scaler is not None:
        # Show both original and normalized metrics
        if 'mae_original' in metrics:
            print(f"      Overall MAE (original): {metrics['mae_original']:.2f} cm")
            print(f"      Overall MSE (original): {metrics['mse_original']:.2f} cm²")
        if 'final_step_mae_original' in metrics:
            print(f"      Final step MAE (original): {metrics['final_step_mae_original']:.2f} cm")
            print(f"      Final step MSE (original): {metrics['final_step_mse_original']:.2f} cm²")
    
    # Always show normalized metrics
    print(f"      Overall MAE (normalized): {metrics['mae_normalized']:.4f}")
    print(f"      Overall MSE (normalized): {metrics['mse_normalized']:.4f}")
    print(f"      Final step MAE (normalized): {metrics['final_step_mae_normalized']:.4f}")
    print(f"      Final step MSE (normalized): {metrics['final_step_mse_normalized']:.4f}")
    print(f"      Overall R²: {metrics['r2_normalized']:.4f}")
    print(f"      Trajectory MAE (avg): {metrics['trajectory_mae_normalized']:.4f}")
    
    # Save model and results
    model_filename = f"{model_name}_multistep_{pred_len}step"
    save_model_and_results(
        model, history, metrics, model_filename,
        DATA_PATHS['trained_models']
    )
    
    # Create visualizations
    plot_multistep_predictions(
        predictions, targets, f"{model_name}_multistep", 
        DATA_PATHS['evaluation_plots'], position_scaler
    )
    
    return model, history, metrics

def main():
    """Main training function"""
    print("=" * 80)
    print("MULTI-STEP REGRESSION MODEL TRAINING")
    print("=" * 80)
    
    # Models to train for multi-step prediction
    models_to_train = ['lstm', 'gru']
    pred_len = 30
    
    results = {}
    
    for model_name in models_to_train:
        try:
            model, history, metrics = train_multistep_model(
                model_name, 
                pred_len=pred_len,
                num_epochs=100,
                batch_size=16,  # Smaller batch size for multi-step
                patience=20
            )
            
            results[model_name] = {
                'model': model,
                'history': history,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-STEP TRAINING SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"{'Model':<8} {'Parameters':<12} {'Final MAE (cm)':<16} {'Final MSE (cm²)':<16} {'Overall R²':<12}")
        print("-" * 80)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            param_count = count_parameters(result['model'])
            
            # Get original scale metrics if available
            position_scaler = load_scaler(None, 'regression', 'position')
            if position_scaler is not None and 'final_step_mae_original' in metrics:
                mae_final_cm = metrics['final_step_mae_original']
                mse_final_cm = metrics['final_step_mse_original']
            else:
                # Fallback to normalized metrics converted to cm scale
                mae_final_cm = metrics['final_step_mae_normalized']
                mse_final_cm = metrics['final_step_mse_normalized']
                if position_scaler is not None:
                    mae_final_cm *= position_scaler.scale_[0]
                    mse_final_cm *= (position_scaler.scale_[0] ** 2)
            
            print(f"{model_name.upper():<8} "
                  f"{param_count:<12,d} "
                  f"{mae_final_cm:<16.2f} "
                  f"{mse_final_cm:<16.2f} "
                  f"{metrics['r2_normalized']:<12.4f}")
        
        # Find best model based on final step MAE
        best_model = min(results.keys(), key=lambda x: results[x]['metrics']['final_step_mae_normalized'])
        best_mae = results[best_model]['metrics']['final_step_mae_normalized']
        position_scaler = load_scaler(None, 'regression', 'position')
        if position_scaler is not None:
            best_mae_cm = best_mae * position_scaler.scale_[0]
            print(f"\n🏆 Best model: {best_model.upper()} (Final step MAE: {best_mae_cm:.2f} cm)")
        else:
            print(f"\n🏆 Best model: {best_model.upper()} (Final step MAE: {best_mae:.4f} normalized)")
    else:
        print("❌ No models were successfully trained.")
    
    print(f"\n✅ Multi-step training completed!")
    print(f"📊 Results saved in: {DATA_PATHS['trained_models']}/")
    print(f"📊 Plots saved in: {DATA_PATHS['evaluation_plots']}/")

if __name__ == "__main__":
    main()