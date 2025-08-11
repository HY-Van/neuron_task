"""
Regression Model Evaluation Script
Evaluates trained regression models on test data
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import utils with centralized configuration
from utils import (
    DATA_PATHS, DATASET_CONFIG,
    load_processed_data, create_data_loaders, load_scaler,
    calculate_regression_metrics, count_parameters
)

# Import all models
from models.RNN import get_rnn_model
from models.LSTM import get_lstm_model
from models.GRU import get_gru_model
from models.TCN import get_tcn_model
from models.LTSF_Linear import get_ltsf_linear_model
from models.RWKV import get_rwkv_model

def load_trained_model(model_name, input_size, output_size, device):
    """Load a trained regression model"""
    
    # Create model
    model_configs = {
        'rnn': {'task_type': 'regression', 'input_size': input_size, 'output_size': output_size},
        'lstm': {'task_type': 'regression', 'input_size': input_size, 'output_size': output_size},
        'gru': {'task_type': 'regression', 'input_size': input_size, 'output_size': output_size},
        'tcn': {'input_size': input_size, 'output_size': output_size},
        'ltsf_linear': {'seq_len': DATASET_CONFIG['window_size'], 'enc_in': input_size, 'output_size': output_size},
        'rwkv': {'input_size': input_size, 'output_size': output_size}
    }
    
    model_functions = {
        'rnn': get_rnn_model,
        'lstm': get_lstm_model,
        'gru': get_gru_model,
        'tcn': get_tcn_model,
        'ltsf_linear': get_ltsf_linear_model,
        'rwkv': get_rwkv_model
    }
    
    # Create model with correct parameters
    if model_name in ['rnn', 'lstm', 'gru']:
        model = model_functions[model_name](**model_configs[model_name])
    else:
        # For TCN, LTSF_Linear, RWKV, pass 'regression' as first argument
        model = model_functions[model_name]('regression', **model_configs[model_name])
    
    model.to(device)
    
    # Load weights
    model_path = f"{DATA_PATHS['trained_models']}/{model_name}_regression.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device, position_scaler=None):
    """Evaluate model on test data"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(targets, predictions, position_scaler)
    
    return predictions, targets, metrics

def plot_predictions(predictions, targets, model_name, save_dir, position_scaler=None):
    """Plot prediction vs actual scatter plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use original scale if scaler available
    if position_scaler is not None:
        pred_plot = position_scaler.inverse_transform(predictions)
        targ_plot = position_scaler.inverse_transform(targets)
        unit = "cm"
    else:
        pred_plot = predictions
        targ_plot = targets
        unit = "(normalized)"
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # X coordinate
    axes[0].scatter(targ_plot[:, 0], pred_plot[:, 0], alpha=0.6, s=1)
    axes[0].plot([targ_plot[:, 0].min(), targ_plot[:, 0].max()], 
                 [targ_plot[:, 0].min(), targ_plot[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel(f'True X Position {unit}')
    axes[0].set_ylabel(f'Predicted X Position {unit}')
    axes[0].set_title(f'{model_name.upper()} - X Coordinate Predictions')
    axes[0].grid(True, alpha=0.3)
    
    # Y coordinate
    axes[1].scatter(targ_plot[:, 1], pred_plot[:, 1], alpha=0.6, s=1)
    axes[1].plot([targ_plot[:, 1].min(), targ_plot[:, 1].max()], 
                 [targ_plot[:, 1].min(), targ_plot[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel(f'True Y Position {unit}')
    axes[1].set_ylabel(f'Predicted Y Position {unit}')
    axes[1].set_title(f'{model_name.upper()} - Y Coordinate Predictions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"{save_dir}/{model_name}_regression_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Prediction plots saved: {save_path}")

def plot_2d_trajectory(predictions, targets, model_name, save_dir, position_scaler=None, max_samples=1000):
    """Plot 2D trajectory comparison"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use original scale if scaler available
    if position_scaler is not None:
        pred_plot = position_scaler.inverse_transform(predictions)
        targ_plot = position_scaler.inverse_transform(targets)
        unit = "cm"
    else:
        pred_plot = predictions
        targ_plot = targets
        unit = "(normalized)"
    
    # Subsample for visualization if too many points
    if len(pred_plot) > max_samples:
        indices = np.random.choice(len(pred_plot), max_samples, replace=False)
        pred_sub = pred_plot[indices]
        targ_sub = targ_plot[indices]
    else:
        pred_sub = pred_plot
        targ_sub = targ_plot
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(targ_sub[:, 0], targ_sub[:, 1], 
               alpha=0.6, s=20, label='True Trajectory', c='blue')
    plt.scatter(pred_sub[:, 0], pred_sub[:, 1], 
               alpha=0.6, s=20, label='Predicted Trajectory', c='red')
    
    plt.xlabel(f'X Position {unit}')
    plt.ylabel(f'Y Position {unit}')
    plt.title(f'{model_name.upper()} - 2D Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    save_path = f"{save_dir}/{model_name}_2d_trajectory.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 2D trajectory plot saved: {save_path}")

def plot_model_comparison(results, save_dir):
    """Create comparison plots for all models"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    
    # Check if we have original scale metrics
    has_original = any('mae_original' in results[model]['metrics'] for model in models)
    
    if has_original:
        # Extract original scale metrics (priority)
        mae_orig = [results[model]['metrics']['mae_original'] for model in models]
        mse_orig = [results[model]['metrics']['mse_original'] for model in models]
        r2_orig = [results[model]['metrics']['r2_original'] for model in models]
        
        # Create comparison plot with original scale metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].bar(models, mae_orig, color=['darkblue', 'darkgreen', 'darkred', 'orange', 'purple', 'brown'][:len(models)])
        axes[0].set_title('Mean Absolute Error (Original Scale)', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('MAE (cm)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(models, mse_orig, color=['darkblue', 'darkgreen', 'darkred', 'orange', 'purple', 'brown'][:len(models)])
        axes[1].set_title('Mean Squared Error (Original Scale)', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('MSE (cm²)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(models, r2_orig, color=['darkblue', 'darkgreen', 'darkred', 'orange', 'purple', 'brown'][:len(models)])
        axes[2].set_title('R² Score (Original Scale)', fontweight='bold', fontsize=14)
        axes[2].set_ylabel('R²', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
    else:
        # Use normalized metrics
        mae_norm = [results[model]['metrics']['mae_normalized'] for model in models]
        mse_norm = [results[model]['metrics']['mse_normalized'] for model in models]
        r2_norm = [results[model]['metrics']['r2_normalized'] for model in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].bar(models, mae_norm, color=['lightblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown'][:len(models)])
        axes[0].set_title('Mean Absolute Error (Normalized)', fontsize=14)
        axes[0].set_ylabel('MAE')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(models, mse_norm, color=['lightblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown'][:len(models)])
        axes[1].set_title('Mean Squared Error (Normalized)', fontsize=14)
        axes[1].set_ylabel('MSE')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(models, r2_norm, color=['lightblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown'][:len(models)])
        axes[2].set_title('R² Score (Normalized)', fontsize=14)
        axes[2].set_ylabel('R²')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    save_path = f"{save_dir}/regression_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison plot saved: {save_path}")

def main():
    """Main evaluation function"""
    print("=" * 80)
    print("EVALUATING REGRESSION MODELS")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load test data
    print(f"\n📁 Loading test data from {DATA_PATHS['processed_data']}...")
    data_dict, metadata = load_processed_data(None, 'regression')
    
    _, _, test_loader = create_data_loaders(data_dict, 'regression', batch_size=64)
    
    print(f"   Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   Output dimensions: {data_dict['y_test'].shape[1]} (x, y coordinates)")
    
    # Load position scaler
    position_scaler = load_scaler(None, 'regression', 'position')
    if position_scaler is not None:
        print("   ✅ Position scaler loaded for original scale metrics")
    else:
        print("   ⚠️  Position scaler not found - using normalized metrics only")
    
    # Models to evaluate
    models_to_evaluate = ['rnn', 'lstm', 'gru', 'tcn', 'ltsf_linear', 'rwkv']
    results = {}
    
    input_size = data_dict['X_train'].shape[2]
    output_size = data_dict['y_train'].shape[1]
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"\n🔍 Evaluating {model_name.upper()} model...")
        
        try:
            model = load_trained_model(model_name, input_size, output_size, device)
            print(f"   ✅ Model loaded successfully")
            
            predictions, targets, metrics = evaluate_model(model, test_loader, device, position_scaler)
            
            results[model_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets,
                'param_count': count_parameters(model)
            }
            
            # Print metrics - original scale first if available
            print(f"   📊 Results:")
            if position_scaler is not None and 'mae_original' in metrics:
                print(f"      MAE (original):   {metrics['mae_original']:.2f} cm")
                print(f"      MSE (original):   {metrics['mse_original']:.2f} cm²")
                print(f"      R² (original):    {metrics['r2_original']:.4f}")
            print(f"      MAE (normalized): {metrics['mae_normalized']:.4f}")
            print(f"      R² (normalized):  {metrics['r2_normalized']:.4f}")
            
            # Create plots
            plot_predictions(predictions, targets, model_name, DATA_PATHS['evaluation_plots'], position_scaler)
            plot_2d_trajectory(predictions, targets, model_name, DATA_PATHS['evaluation_plots'], position_scaler)
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    # Save results
    if results:
        results_path = f"{DATA_PATHS['trained_models']}/regression_evaluation_results.json"
        serializable_results = {}
        
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'metrics': {k: float(v) for k, v in model_results['metrics'].items() if not k.endswith('_original') or v is not None},
                'param_count': int(model_results['param_count'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n📋 Results saved: {results_path}")
        
        # Create comparison plots
        plot_model_comparison(results, DATA_PATHS['evaluation_plots'])
    
    # Print summary
    print("\n" + "=" * 80)
    print("REGRESSION EVALUATION SUMMARY")
    print("=" * 80)
    
    if results:
        # Determine which metrics to show
        has_original = any('mae_original' in results[model]['metrics'] and results[model]['metrics']['mae_original'] is not None for model in results)
        
        if has_original:
            print(f"{'Model':<12} {'MAE (cm)':<10} {'MSE (cm²)':<12} {'R² (orig)':<10} {'Parameters':<12}")
            print("-" * 70)
            
            for model_name, model_results in results.items():
                metrics = model_results['metrics']
                param_count = model_results['param_count']
                print(f"{model_name.upper():<12} "
                      f"{metrics.get('mae_original', 0):<10.2f} "
                      f"{metrics.get('mse_original', 0):<12.2f} "
                      f"{metrics.get('r2_original', 0):<10.4f} "
                      f"{param_count:<12,d}")
            
            # Find best model based on original scale MAE
            best_model = min(results.keys(), key=lambda x: results[x]['metrics'].get('mae_original', float('inf')))
            best_mae = results[best_model]['metrics']['mae_original']
            print(f"\n🏆 Best model: {best_model.upper()} (MAE: {best_mae:.2f} cm)")
        else:
            print(f"{'Model':<12} {'MAE (norm)':<12} {'R² (norm)':<10} {'Parameters':<12}")
            print("-" * 60)
            
            for model_name, model_results in results.items():
                metrics = model_results['metrics']
                param_count = model_results['param_count']
                print(f"{model_name.upper():<12} "
                      f"{metrics['mae_normalized']:<12.4f} "
                      f"{metrics['r2_normalized']:<10.4f} "
                      f"{param_count:<12,d}")
            
            # Find best model based on normalized MAE
            best_model = min(results.keys(), key=lambda x: results[x]['metrics']['mae_normalized'])
            best_mae = results[best_model]['metrics']['mae_normalized']
            print(f"\n🏆 Best model: {best_model.upper()} (MAE: {best_mae:.4f} normalized)")
    else:
        print("❌ No models were successfully evaluated.")
    
    print(f"\n✅ Regression evaluation completed!")
    print(f"📊 Plots saved in: {DATA_PATHS['evaluation_plots']}/")

if __name__ == "__main__":
    main()
