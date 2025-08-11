"""
Classification Model Evaluation Script
Evaluates trained classification models on test data
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import utils with centralized configuration
from utils import (
    DATA_PATHS, DATASET_CONFIG,
    load_processed_data, create_data_loaders,
    calculate_classification_metrics, count_parameters
)

# Import all models
from models.RNN import get_rnn_model
from models.LSTM import get_lstm_model
from models.GRU import get_gru_model
from models.TCN import get_tcn_model
from models.LTSF_Linear import get_ltsf_linear_model
from models.RWKV import get_rwkv_model

def load_trained_model(model_name, input_size, num_classes, device):
    """Load a trained classification model"""
    
    # Create model
    model_configs = {
        'rnn': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'lstm': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'gru': {'task_type': 'classification', 'input_size': input_size, 'num_classes': num_classes},
        'tcn': {'input_size': input_size, 'num_classes': num_classes},
        'ltsf_linear': {'seq_len': DATASET_CONFIG['window_size'], 'enc_in': input_size, 'num_classes': num_classes},
        'rwkv': {'input_size': input_size, 'num_classes': num_classes}
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
        # For TCN, LTSF_Linear, RWKV, pass 'classification' as first argument
        model = model_functions[model_name]('classification', **model_configs[model_name])
    
    model.to(device)
    
    # Load weights
    model_path = f"{DATA_PATHS['trained_models']}/{model_name}_classification.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(y_batch.cpu().numpy())
    
    return np.array(all_true_labels), np.array(all_predictions)

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Plot confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    save_path = f"{save_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Confusion matrix saved: {save_path}")

def plot_model_comparison(results, save_dir):
    """Create comparison plots for all models"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'brown']
    
    for i, metric in enumerate(metrics):
        values = [results[model]['metrics'][metric] for model in models]
        bars = axes[i].bar(models, values, color=colors[:len(models)], alpha=0.8)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = f"{save_dir}/classification_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison plot saved: {save_path}")

def main():
    """Main evaluation function"""
    print("=" * 80)
    print("EVALUATING CLASSIFICATION MODELS")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load test data
    print(f"\n📁 Loading test data from {DATA_PATHS['processed_data']}...")
    data_dict, metadata = load_processed_data(None, 'classification')
    
    _, _, test_loader = create_data_loaders(data_dict, 'classification', batch_size=64)
    
    print(f"   Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   Number of classes: {metadata['n_classes']}")
    
    # Models to evaluate
    models_to_evaluate = ['rnn', 'lstm', 'gru', 'tcn', 'ltsf_linear', 'rwkv']
    results = {}
    
    input_size = data_dict['X_train'].shape[2]
    num_classes = metadata['n_classes']
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"\n🔍 Evaluating {model_name.upper()} model...")
        
        try:
            model = load_trained_model(model_name, input_size, num_classes, device)
            print(f"   ✅ Model loaded successfully")
            
            y_true, y_pred = evaluate_model(model, test_loader, device)
            metrics = calculate_classification_metrics(y_true, y_pred)
            
            results[model_name] = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'param_count': count_parameters(model)
            }
            
            print(f"   📊 Results:")
            print(f"      Accuracy:  {metrics['accuracy']:.4f}")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall:    {metrics['recall']:.4f}")
            print(f"      F1-Score:  {metrics['f1_score']:.4f}")
            
            # Create confusion matrix
            plot_confusion_matrix(y_true, y_pred, model_name, DATA_PATHS['evaluation_plots'])
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    # Save results
    if results:
        results_path = f"{DATA_PATHS['trained_models']}/classification_evaluation_results.json"
        serializable_results = {}
        
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'metrics': {k: float(v) for k, v in model_results['metrics'].items()},
                'param_count': int(model_results['param_count'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n📋 Results saved: {results_path}")
        
        # Create comparison plots
        plot_model_comparison(results, DATA_PATHS['evaluation_plots'])
    
    # Print summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION EVALUATION SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Parameters':<12}")
        print("-" * 80)
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            param_count = model_results['param_count']
            print(f"{model_name.upper():<12} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{param_count:<12,d}")
        
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])
        best_acc = results[best_model]['metrics']['accuracy']
        
        print(f"\n🏆 Best model: {best_model.upper()} (Accuracy: {best_acc:.4f})")
    else:
        print("❌ No models were successfully evaluated.")
    
    print(f"\n✅ Classification evaluation completed!")
    print(f"📊 Plots saved in: {DATA_PATHS['evaluation_plots']}/")

if __name__ == "__main__":
    main()
