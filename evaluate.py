"""
Evaluation script for classification models
Evaluates trained Simple RNN, LSTM, and GRU models
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)

from models.classification_model import get_model
from train_classification import NeuronDataset, load_processed_data, create_data_loaders

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_trained_model(model_name: str, 
                      model_dir: str = 'trained_models',
                      device: torch.device = None) -> nn.Module:
    """
    Load a trained model
    
    Args:
        model_name: Name of the model ('rnn', 'lstm', 'gru')
        model_dir: Directory containing saved models
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = get_model(model_name)
    
    # Load state dict
    model_path = f"{model_dir}/{model_name}_classifier.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model: nn.Module, 
                  test_loader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Tuple of (true_labels, predictions, probabilities)
    """
    model.eval()
    
    all_true = []
    all_pred = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_true.extend(y_batch.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    return np.array(all_true), np.array(all_pred), all_probs


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         model_name: str,
                         save_dir: str = 'evaluation_plots') -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(25), yticklabels=range(25))
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    # Save plot
    save_path = f"{save_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Confusion matrix saved: {save_path}")


def plot_class_performance(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          model_name: str,
                          save_dir: str = 'evaluation_plots') -> None:
    """
    Plot per-class performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extract per-class metrics
    classes = sorted([int(k) for k in report.keys() if k.isdigit()])
    precision = [report[str(c)]['precision'] for c in classes]
    recall = [report[str(c)]['recall'] for c in classes]
    f1 = [report[str(c)]['f1-score'] for c in classes]
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Precision
    axes[0].bar(classes, precision, alpha=0.7, color='skyblue')
    axes[0].set_title(f'Precision per Class - {model_name.upper()}')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].bar(classes, recall, alpha=0.7, color='lightcoral')
    axes[1].set_title(f'Recall per Class - {model_name.upper()}')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    # F1-score
    axes[2].bar(classes, f1, alpha=0.7, color='lightgreen')
    axes[2].set_title(f'F1-Score per Class - {model_name.upper()}')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = f"{save_dir}/{model_name}_class_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Class performance plot saved: {save_path}")


def compare_models(results: Dict[str, Dict[str, Any]], 
                  save_dir: str = 'evaluation_plots') -> None:
    """
    Create comparison plots for all models
    
    Args:
        results: Dictionary containing results for all models
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics for comparison
    models = list(results.keys())
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    # Create comparison data
    comparison_data = {metric: [results[model]['metrics'][metric] for model in models] 
                      for metric in metrics}
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, (metric, values) in enumerate(comparison_data.items()):
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([model.upper() for model in models])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (metric, values) in enumerate(comparison_data.items()):
        for j, value in enumerate(values):
            ax.text(j + i * width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = f"{save_dir}/model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison plot saved: {save_path}")


def save_evaluation_results(results: Dict[str, Dict[str, Any]], 
                           save_path: str = 'evaluation_results.json') -> None:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary containing all evaluation results
        save_path: Path to save results
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'metrics': model_results['metrics'],
            'classification_report': model_results['classification_report']
        }
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📋 Evaluation results saved: {save_path}")


def main():
    """
    Main evaluation function
    """
    print("=" * 80)
    print("EVALUATING CLASSIFICATION MODELS")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load test data
    print("\n📁 Loading test data...")
    data_dict, metadata = load_processed_data()
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(data_dict, batch_size=64)
    
    print(f"   Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   Number of classes: {metadata['n_classes']}")
    
    # Models to evaluate
    models_to_evaluate = ['rnn', 'lstm', 'gru']
    results = {}
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"\n🔍 Evaluating {model_name.upper()} model...")
        
        try:
            # Load trained model
            model = load_trained_model(model_name, device=device)
            print(f"   ✅ Model loaded successfully")
            
            # Evaluate on test data
            y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred)
            
            # Get detailed classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'classification_report': class_report,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_probs': y_probs
            }
            
            # Print metrics
            print(f"   📊 Results:")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            print(f"      Precision (weighted): {metrics['precision_weighted']:.4f}")
            print(f"      Recall (weighted): {metrics['recall_weighted']:.4f}")
            print(f"      F1-score (weighted): {metrics['f1_weighted']:.4f}")
            
            # Create visualizations
            plot_confusion_matrix(y_true, y_pred, model_name)
            plot_class_performance(y_true, y_pred, model_name)
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    # Compare models
    if len(results) > 1:
        print(f"\n📊 Creating model comparison...")
        compare_models(results)
    
    # Save results
    save_evaluation_results(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"{'Model':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            print(f"{model_name.upper():<8} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision_weighted']:<10.4f} "
                  f"{metrics['recall_weighted']:<10.4f} "
                  f"{metrics['f1_weighted']:<10.4f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])
        best_accuracy = results[best_model]['metrics']['accuracy']
        
        print(f"\n🏆 Best model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
        
    else:
        print("❌ No models were successfully evaluated.")
    
    print(f"\n✅ Evaluation completed!")
    print("   📊 Plots saved in 'evaluation_plots/' directory")
    print("   📋 Results saved in 'evaluation_results.json'")


if __name__ == "__main__":
    main()
