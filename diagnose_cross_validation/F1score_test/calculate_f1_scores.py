#!/usr/bin/env python3
"""
F1-Score Calculation for Cross-validation Models
Author: AI Assistant
Date: 2024-09-25
Purpose: Calculate precision, recall, F1-score for each cell type across both validation batches
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
import os
import argparse

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Net(nn.Module):
    """Neural Network Model Architecture"""
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=6, activation=nn.ReLU()):
        super(Net, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size, num_classes))
        self.activation = activation
    
    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)
        x = self.fc[-1](x)
        return x

def load_and_prepare_data(data_path):
    """Load and prepare data for evaluation"""
    print(f"Loading data from: {data_path}")
    original_test_data = pd.read_csv(data_path, index_col=0)
    
    # Extract gene names
    feature_names = original_test_data.index.tolist()
    
    # Reset index and transpose
    data = original_test_data.reset_index(drop=True)
    data = data.transpose()
    
    # Extract labels (cell types)
    labels = data.index
    # Important: R matrix output adds sequence numbers to identical labels
    labels = [label.split('.')[0] for label in data.index]
    
    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Prepare features
    features = data.values
    
    # Convert to tensors
    labels_tensor = torch.tensor(encoded_labels, device=device).long()
    features_tensor = torch.tensor(features.astype(float), device=device)
    
    # Split data (using all for testing as in original code)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_tensor, labels_tensor, 
        train_size=None,
        random_state=42
    )
    
    # Create test dataset
    test_dataset = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return test_loader, encoder, features_tensor.shape[1], len(np.unique(encoded_labels))

def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    model.to(device)
    
    y_pred, y_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device).float(), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def calculate_metrics(y_true, y_pred, encoder, batch_name):
    """Calculate detailed metrics for each class"""
    # Get class names
    class_names = encoder.classes_
    
    # Calculate precision, recall, F1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Calculate overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'Cell_Type': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Support': support
    })
    
    # Add overall metrics
    overall_metrics = {
        'Batch': batch_name,
        'Overall_Accuracy': accuracy,
        'Macro_Precision': precision_macro,
        'Macro_Recall': recall_macro,
        'Macro_F1': f1_macro,
        'Weighted_Precision': precision_weighted,
        'Weighted_Recall': recall_weighted,
        'Weighted_F1': f1_weighted
    }
    
    return results_df, overall_metrics

def create_detailed_report(batch_configs, output_dir):
    """Create F1-score reports for both batches using provided config and output dir"""
    all_results = []
    overall_results = []
    
    for batch_name, config in batch_configs.items():
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_name.upper()}")
        print(f"{'='*50}")
        
        # Load and prepare data
        test_loader, encoder, input_size, num_classes = load_and_prepare_data(config['data_path'])
        
        # Initialize and load model
        model = Net(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_classes=num_classes,
            num_hidden_layers=config['hidden_layers'],
            activation=config['activation']
        )
        
        print(f"Loading model from: {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        model.to(device)
        
        # Evaluate model
        y_true, y_pred = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        results_df, overall_metrics = calculate_metrics(y_true, y_pred, encoder, batch_name)
        
        # Add batch info to detailed results
        results_df['Batch'] = batch_name.upper()
        
        # Store results
        all_results.append(results_df)
        overall_results.append(overall_metrics)
        
        # Print summary
        print(f"Accuracy: {overall_metrics['Overall_Accuracy']:.4f} ({overall_metrics['Overall_Accuracy']*100:.2f}%)")
        print(f"Macro F1-Score: {overall_metrics['Macro_F1']:.4f}")
        print(f"Weighted F1-Score: {overall_metrics['Weighted_F1']:.4f}")
        
        # Save individual batch results
        batch_output_path = os.path.join(output_dir, f'{batch_name}_detailed_metrics.csv')
        results_df.to_csv(batch_output_path, index=False)
        print(f"Detailed metrics saved to: {batch_output_path}")
        
        # Create confusion matrix
        create_confusion_matrix_plot(y_true, y_pred, encoder.classes_, batch_name, output_dir)
    
    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_output_path = os.path.join(output_dir, 'combined_detailed_metrics.csv')
    combined_results.to_csv(combined_output_path, index=False)
    
    # Create overall summary
    overall_df = pd.DataFrame(overall_results)
    overall_output_path = os.path.join(output_dir, 'overall_summary_metrics.csv')
    overall_df.to_csv(overall_output_path, index=False)
    
    # Create visualization
    create_performance_visualization(combined_results, overall_df, output_dir)
    
    print(f"\n{'='*50}")
    print("SUMMARY COMPLETED")
    print(f"{'='*50}")
    print(f"Combined detailed metrics: {combined_output_path}")
    print(f"Overall summary metrics: {overall_output_path}")
    print(f"All outputs saved to: {output_dir}")
    
    return combined_results, overall_df

def create_confusion_matrix_plot(y_true, y_pred, class_names, batch_name, output_dir):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Batch {batch_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{batch_name}_confusion_matrix_detailed.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    plot_path_png = os.path.join(output_dir, f'{batch_name}_confusion_matrix_detailed.png')
    plt.savefig(plot_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved: {plot_path}")

def create_performance_visualization(combined_results, overall_df, output_dir):
    """Create comprehensive performance visualization"""
    
    # 1. F1-Score comparison by cell type
    plt.figure(figsize=(15, 8))
    
    # Create pivot table for plotting
    pivot_data = combined_results.pivot(index='Cell_Type', columns='Batch', values='F1_Score')
    
    # Create grouped bar plot
    ax = pivot_data.plot(kind='bar', width=0.8, figsize=(15, 8))
    plt.title('F1-Score by Cell Type and Batch', fontsize=16, fontweight='bold')
    plt.xlabel('Cell Type', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend(title='Batch', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    f1_plot_path = os.path.join(output_dir, 'f1_scores_by_celltype.pdf')
    plt.savefig(f1_plot_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'f1_scores_by_celltype.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Overall metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = ['Overall_Accuracy', 'Macro_F1', 'Weighted_F1', 'Macro_Precision']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        bars = ax.bar(overall_df['Batch'], overall_df[metric], color=colors[i], alpha=0.8)
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    overall_plot_path = os.path.join(output_dir, 'overall_metrics_comparison.pdf')
    plt.savefig(overall_plot_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Performance visualizations saved:")
    print(f"  - F1-scores by cell type: {f1_plot_path}")
    print(f"  - Overall metrics: {overall_plot_path}")

def create_manuscript_table(combined_results, overall_df, output_dir):
    """Create publication-ready table for manuscript"""
    
    # Calculate summary statistics for manuscript
    summary_stats = []
    
    for batch in ['b1', 'b2']:
        batch_data = combined_results[combined_results['Batch'] == batch.upper()]
        batch_overall = overall_df[overall_df['Batch'] == batch]
        
        # Get min, max, mean F1-scores
        f1_scores = batch_data['F1_Score'].values
        f1_mean = np.mean(f1_scores)
        f1_min = np.min(f1_scores)
        f1_max = np.max(f1_scores)
        
        accuracy = batch_overall['Overall_Accuracy'].iloc[0]
        
        summary_stats.append({
            'Batch': batch.upper(),
            'Accuracy': f"{accuracy:.4f} ({accuracy*100:.2f}%)",
            'F1_Range': f"{f1_min:.3f}-{f1_max:.3f}",
            'Mean_F1': f"{f1_mean:.3f}",
            'Cell_Types': len(batch_data)
        })
    
    manuscript_df = pd.DataFrame(summary_stats)
    manuscript_path = os.path.join(output_dir, 'manuscript_summary_table.csv')
    manuscript_df.to_csv(manuscript_path, index=False)
    
    print(f"Manuscript summary table saved: {manuscript_path}")
    
    return manuscript_df

if __name__ == "__main__":
    print("Starting F1-Score Analysis for Cross-validation Models...")
    print("This script will calculate precision, recall, and F1-score for each cell type.")

    parser = argparse.ArgumentParser(description="Calculate F1-scores and generate figures/tables.")
    parser.add_argument("--b1-data", dest="b1_data", type=str, required=True,
                        help="Path to Batch1 evaluation CSV (tested by b2 model).")
    parser.add_argument("--b2-data", dest="b2_data", type=str, required=True,
                        help="Path to Batch2 evaluation CSV (tested by b1 model).")
    parser.add_argument("--b1-model", dest="b1_model", type=str, required=True,
                        help="Path to trained Batch1 model .pth file.")
    parser.add_argument("--b2-model", dest="b2_model", type=str, required=True,
                        help="Path to trained Batch2 model .pth file.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=os.getcwd(),
                        help="Directory to write outputs (default: CWD).")
    parser.add_argument("--b1-hidden-layers", dest="b1_hidden_layers", type=int, default=4)
    parser.add_argument("--b1-hidden-size", dest="b1_hidden_size", type=int, default=64)
    parser.add_argument("--b2-hidden-layers", dest="b2_hidden_layers", type=int, default=5)
    parser.add_argument("--b2-hidden-size", dest="b2_hidden_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    batch_configs = {
        'b1': {
            'data_path': args.b2_data,
            'model_path': args.b1_model,
            'hidden_layers': args.b1_hidden_layers,
            'hidden_size': args.b1_hidden_size,
            'activation': nn.ELU()
        },
        'b2': {
            'data_path': args.b1_data,
            'model_path': args.b2_model,
            'hidden_layers': args.b2_hidden_layers,
            'hidden_size': args.b2_hidden_size,
            'activation': nn.ELU()
        }
    }

    combined_results, overall_df = create_detailed_report(batch_configs, args.output_dir)

    manuscript_table = create_manuscript_table(combined_results, overall_df, args.output_dir)

    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*50}")
    print("\nKey Results:")
    print(overall_df[['Batch', 'Overall_Accuracy', 'Macro_F1', 'Weighted_F1']].to_string(index=False))
    print(f"\nAll files saved to: {args.output_dir}")
