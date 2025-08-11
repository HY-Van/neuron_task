"""
Classification Model Training Script
Simplified and unified training for all classification models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from tqdm import tqdm

# Import utils with centralized configuration
from utils import (
    DATA_PATHS, DATASET_CONFIG, 
    load_processed_data, create_data_loaders,
    calculate_classification_metrics, save_model_and_results,
    count_parameters
)

# Import all models
from models.RNN import get_rnn_model
from models.LSTM import get_lstm_model
from models.GRU import get_gru_model
from models.TCN import get_tcn_model
from models.LTSF_Linear import get_ltsf_linear_model
from models.RWKV import get_rwkv_model

warnings.filterwarnings('ignore')

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
        # Update progress bar every 50 batches
        if batch_idx % 50 == 0:
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate_model(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def create_model(model_name, input_size, num_classes):
    """Create model based on name"""
    model_configs = {
        'rnn': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'lstm': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'gru': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'tcn': {'input_size': input_size, 'num_classes': num_classes},  # TCN uses different parameter names
        'ltsf_linear': {'seq_len': DATASET_CONFIG['window_size'], 'enc_in': input_size, 'num_classes': num_classes},  # LTSF uses different parameter names
        'rwkv': {'input_size': input_size, 'num_classes': num_classes}  # RWKV uses different parameter names
    }
    
    model_functions = {
        'rnn': get_rnn_model,
        'lstm': get_lstm_model,
        'gru': get_gru_model,
        'tcn': get_tcn_model,
        'ltsf_linear': get_ltsf_linear_model,
        'rwkv': get_rwkv_model
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Add task_type for models that need it
    if model_name in ['rnn', 'lstm', 'gru']:
        return model_functions[model_name](**model_configs[model_name])
    else:
        # For TCN, LTSF_Linear, RWKV, pass 'classification' as first argument
        return model_functions[model_name]('classification', **model_configs[model_name])

def train_model(model_name, data_dir=None, num_epochs=50, batch_size=64, patience=15):
    """Train a single model"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print(f"\n📁 Loading data from {data_dir or DATA_PATHS['processed_data']}...")
    data_dict, metadata = load_processed_data(data_dir, 'classification')
    
    print(f"   Training samples: {data_dict['X_train'].shape[0]:,}")
    print(f"   Validation samples: {data_dict['X_val'].shape[0]:,}")
    print(f"   Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   Input shape: {data_dict['X_train'].shape[1:]}")
    print(f"   Number of classes: {metadata['n_classes']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict, 'classification', batch_size
    )
    
    # Create model
    input_size = data_dict['X_train'].shape[2]  # Feature dimension
    num_classes = metadata['n_classes']
    
    print(f"\n🔧 Creating {model_name.upper()} model...")
    model = create_model(model_name, input_size, num_classes)
    model.to(device)
    
    param_count = count_parameters(model)
    print(f"   Parameters: {param_count:,}")
    
    # Setup training with improved optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    
    # Model-specific learning rates
    model_lrs = {
        'rnn': 0.003,
        'lstm': 0.002, 
        'gru': 0.002,
        'tcn': 0.001,
        'ltsf_linear': 0.005,
        'rwkv': 0.001
    }
    
    lr = model_lrs.get(model_name, 0.002)  # Default fallback learning rate
    print(f"   Using learning rate: {lr}")
    
    # AdamW optimizer with different weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5 if param_count < 100000 else 5e-5,  # Adjust weight decay based on model size
        betas=(0.9, 0.95)  # Better betas for convergence
    )
    
    # Cosine Annealing with Warm Restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Initial restart period
        T_mult=2,  # Multiplication factor for restart period
        eta_min=lr * 0.01  # Minimum learning rate
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': []  # Track learning rate changes
    }
    
    # Early stopping with improved patience
    best_val_acc = 0.0  # Track best accuracy instead of loss
    patience_counter = 0
    best_model_state = None
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Update scheduler after each epoch
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {current_lr:.6f}")
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"   🎯 New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏰ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save model and results
    save_model_and_results(model, history, model_name, 'classification', metadata)
    
    return model, history

def main():
    """Main training function"""
    print("=" * 80)
    print("NEURON SIGNAL CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    
    # Models to train
    models_to_train = ['rnn', 'lstm', 'gru', 'tcn', 'ltsf_linear', 'rwkv']
    
    trained_models = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        try:
            model, history = train_model(
                model_name=model_name,
                data_dir=None,  # Use default from DATA_PATHS
                num_epochs=50,
                batch_size=64,
                patience=10
            )
            
            trained_models[model_name] = {
                'model': model,
                'history': history,
                'best_val_acc': max(history['val_accuracy'])
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    if trained_models:
        print(f"{'Model':<12} {'Best Val Acc':<12} {'Parameters':<12}")
        print("-" * 36)
        
        for model_name, results in trained_models.items():
            best_acc = results['best_val_acc']
            param_count = count_parameters(results['model'])
            print(f"{model_name.upper():<12} {best_acc:<12.2f} {param_count:<12,d}")
        
        # Find best model
        best_model_name = max(trained_models.keys(), 
                             key=lambda x: trained_models[x]['best_val_acc'])
        best_acc = trained_models[best_model_name]['best_val_acc']
        print(f"\n🏆 Best model: {best_model_name.upper()} ({best_acc:.2f}%)")
    else:
        print("❌ No models were successfully trained.")
    
    print(f"\n✅ Training completed!")
    print(f"📁 Models saved in: {DATA_PATHS['trained_models']}/")
    print(f"📊 Plots saved in: {DATA_PATHS['evaluation_plots']}/")
    print("   Use evaluate_classification.py to evaluate the models")

if __name__ == "__main__":
    main()
