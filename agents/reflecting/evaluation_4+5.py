import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python evaluation_4+5.py <tunnel_id>")
    print("Example: python evaluation_4+5.py 4-1")
    sys.exit(1)

tunnel_id = sys.argv[1]
input_dir = f"data/{tunnel_id}"

print(f"Starting evaluation for tunnel: {tunnel_id}")

def calculate_metrics(gt_labels, pred_labels, class_names=None):
    """
    Calculate segmentation metrics:
    - Overall Accuracy (OA)
    - F1 Score
    - Mean IoU (mIoU)
    - Per-class IoU
    
    Args:
        gt_labels: Ground truth labels, can be 1D or 2D array
        pred_labels: Predicted labels, can be 1D or 2D array
        class_names: Dictionary mapping class indices to names, defaults to None
    
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are numpy arrays and flatten if needed
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    
    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels
    
    # Get unique classes
    classes = np.unique(np.concatenate((np.unique(gt_flat), np.unique(pred_flat))))
    classes = np.sort(classes)
    
    # Calculate overall accuracy
    oa = accuracy_score(gt_flat, pred_flat)
    
    # Calculate F1 score (macro average)
    f1 = f1_score(gt_flat, pred_flat, average='macro', labels=classes, zero_division=0)
    
    # Calculate IoU for each class
    iou_per_class = jaccard_score(gt_flat, pred_flat, average=None, labels=classes, zero_division=0)
    
    # Calculate mean IoU
    miou = np.mean(iou_per_class)
    
    # Prepare results
    results = {
        'OA': oa,
        'F1': f1,
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'classes': classes
    }
    
    # Print formatted results
    print(f"OA {oa:.3f} F1 {f1:.3f} mIoU {miou:.3f}")
    
    if class_names is None:
        class_names = {}
        for c in classes:
            class_names[c] = f"Class {c}"
    
    print("Per-class IoU:", end=" ")
    class_iou_strs = []
    for i, class_idx in enumerate(classes):
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        class_iou_strs.append(f"{class_name} {iou_per_class[i]:.3f}")
    
    print(", ".join(class_iou_strs))
    
    return results

def plot_confusion_matrices(gt_labels, pred_labels, class_names, present_classes=None, output_file='confusion_matrices.png'):
    """
    Plot both raw and normalized confusion matrices
    
    Args:
        gt_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: Dictionary mapping class indices to names
        present_classes: List of classes present in the data (optional)
        output_file: Name of the output file
    """
    # Ensure inputs are numpy arrays and flatten if needed
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    
    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels
    
    # If present_classes is not provided, determine from the data
    if present_classes is None:
        present_classes = sorted(list(set(np.unique(gt_flat)) | set(np.unique(pred_flat))))
    
    # Create labels for the confusion matrix
    class_labels = [class_names.get(c, f"Class {c}") for c in present_classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=present_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_ylabel('Ground Truth')
    ax1.set_xlabel('Prediction')
    
    # Normalized confusion matrix (row normalization - shows where each ground truth class goes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized by Row)')
    ax2.set_ylabel('Ground Truth')
    ax2.set_xlabel('Prediction')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"\nConfusion matrices saved to '{output_file}'")

def plot_iou_bars(iou_per_class, classes, class_names, output_file='iou_by_class.png'):
    """
    Plot a bar chart of IoU values for each class
    
    Args:
        iou_per_class: Array of IoU values for each class
        classes: Array of class indices
        class_names: Dictionary mapping class indices to names
        output_file: Name of the output file
    """
    # Create class labels
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(class_labels, iou_per_class, color=colors)
    plt.axhline(y=np.mean(iou_per_class), color='r', linestyle='-', label=f'Mean IoU: {np.mean(iou_per_class):.3f}')
    
    # Add values on top of bars
    for bar, iou in zip(bars, iou_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{iou:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Class')
    plt.ylabel('IoU Score')
    plt.title('IoU Scores by Class')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"IoU by class chart saved to '{output_file}'")

def class_distribution_plot(gt_labels, pred_labels, class_names, output_file='class_distribution.png'):
    """
    Plot the distribution of classes in ground truth and predictions (Counts only)
    """
    # Ensure inputs are numpy arrays and flatten if needed
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    
    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels
    
    # Get counts for each class
    classes = sorted(list(set(np.unique(gt_flat)) | set(np.unique(pred_flat))))
    
    gt_counts = np.array([np.sum(gt_flat == c) for c in classes])
    pred_counts = np.array([np.sum(pred_flat == c) for c in classes])
    
    # Create class labels
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    # Create the plot (Counts only)
    plt.figure(figsize=(10, 8))
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, gt_counts, width, label='Ground Truth')
    plt.bar(x + width/2, pred_counts, width, label='Prediction')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution (Counts)')
    plt.xticks(x, class_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # For large numbers, format y-axis with K, M, etc.
    if np.max(gt_counts) > 1000 or np.max(pred_counts) > 1000:
        from matplotlib.ticker import FuncFormatter
        def format_func(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'{x*1e-3:.1f}K'
            else:
                return f'{x:.0f}'
        formatter = FuncFormatter(format_func)
        plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Class distribution (counts) plot saved to '{output_file}'")

def visualize_results(gt_labels, pred_labels, class_names=None, cmap='tab10', output_file='segmentation_comparison.png'):
    """
    Visualize ground truth and prediction labels side by side
    
    Args:
        gt_labels: Ground truth labels, 2D array
        pred_labels: Predicted labels, 2D array
        class_names: Dictionary mapping class indices to names, defaults to None
        cmap: Colormap for visualization, defaults to 'tab10'
        output_file: Name of the output file
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gt_labels, cmap=cmap)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_labels, cmap=cmap)
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Segmentation comparison saved to '{output_file}'")

def generate_example_data(class_names):
    """
    Generate example data for demonstration
    
    Args:
        class_names: List or dictionary of class names
    
    Returns:
        tuple of (ground_truth_labels, predicted_labels)
    """
    np.random.seed(42)  # For reproducibility
    
    # For demonstration, let's create some example data
    h, w = 500, 500
    
    # Create ground truth labels
    gt_labels = np.zeros((h, w), dtype=np.int32)
    
    # Number of classes
    if isinstance(class_names, dict):
        num_classes = len(class_names)
    else:
        num_classes = len(class_names)
    
    # Create regions for each class
    for i in range(num_classes):
        mask = np.random.rand(h, w) < 0.15  # 15% chance of being this class
        gt_labels[mask] = i
    
    # Create predicted labels with some errors
    pred_labels = gt_labels.copy()
    
    # Add some noise (prediction errors)
    noise_mask = np.random.rand(h, w) < 0.12  # 12% noise level
    pred_labels[noise_mask] = np.random.randint(0, num_classes, size=np.sum(noise_mask))
    
    return gt_labels, pred_labels

def evaluate_csv_data():
    """
    Evaluate segmentation performance using data from a CSV file
    """
    # Read the CSV file - using the tunnel_id parameter
    data_path = f"{input_dir}/only_label.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Extract ground truth and predicted labels
    gt_labels = df['gt_labels'].values
    pred_labels = df['pred_labels'].values
    
    # Filter out only classes beyond 7 (class 7 is now included for 7-segment tunnels)
    original_size = len(gt_labels)
    valid_mask = (gt_labels <= 7) & (pred_labels <= 7)
    
    # Log what classes are being filtered out
    gt_beyond_7 = gt_labels[gt_labels > 7]
    pred_beyond_7 = pred_labels[pred_labels > 7]
    if len(gt_beyond_7) > 0 or len(pred_beyond_7) > 0:
        print(f"Filtering out classes beyond 7:")
        for cls in np.unique(np.concatenate([gt_beyond_7, pred_beyond_7])):
            gt_count = np.sum(gt_labels == cls)
            pred_count = np.sum(pred_labels == cls)
            print(f"  Class {cls}: GT={gt_count}, Pred={pred_count} points")
    
    gt_labels = gt_labels[valid_mask]
    pred_labels = pred_labels[valid_mask]
    
    filtered_size = len(gt_labels)
    if original_size != filtered_size:
        print(f"Filtered dataset: {original_size} -> {filtered_size} points ({100*(original_size-filtered_size)/original_size:.2f}% removed)")
    
    # Get unique classes
    unique_gt = np.unique(gt_labels)
    unique_pred = np.unique(pred_labels)
    print(f"Unique ground truth labels: {unique_gt}")
    print(f"Unique predicted labels: {unique_pred}")
    
    # Define class names based on the 7-segment classification system (4+5 version)
    class_names = {
        0: 'Background',
        1: 'K-block',
        2: 'B1-block',
        3: 'A1-block',
        4: 'A2-block',
        5: 'A3-block',
        6: 'A4-block',
        7: 'B2-block'
    }
    
    # Calculate and print metrics
    print(f"\n--- Evaluation Results for {tunnel_id} Data ---")
    results = calculate_metrics(gt_labels, pred_labels, class_names)
    
    # Show class distribution
    print("\nClass distribution:")
    print("Ground truth:")
    for class_idx, name in class_names.items():
        count = np.sum(gt_labels == class_idx)
        percentage = 100 * count / len(gt_labels)
        if count > 0:
            print(f"  {name}: {count} ({percentage:.2f}%)")
    
    print("\nPredictions:")
    for class_idx, name in class_names.items():
        count = np.sum(pred_labels == class_idx)
        percentage = 100 * count / len(pred_labels)
        if count > 0:
            print(f"  {name}: {count} ({percentage:.2f}%)")
    
    # Only include classes that are present in the data
    present_classes = sorted(list(set(np.unique(gt_labels)) | set(np.unique(pred_labels))))
    
    # Make sure output directory exists
    output_dir = f"{input_dir}/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate ONLY the requested visualizations
    print("\nGenerating visualizations...")
    plot_iou_bars(results['IoU_per_class'], results['classes'], class_names, 
                os.path.join(output_dir, 'iou_by_class.png'))
    class_distribution_plot(gt_labels, pred_labels, class_names, 
                          os.path.join(output_dir, 'class_distribution.png'))
    
    # Save performance metrics to performance.md
    markdown_path = os.path.join(output_dir, 'performance.md')
    with open(markdown_path, 'w') as f:
        f.write(f"# Performance Metrics for Tunnel {tunnel_id}\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- Overall Accuracy (OA): {results['OA']:.3f}\n")
        f.write(f"- F1 Score: {results['F1']:.3f}\n")
        f.write(f"- Mean IoU (mIoU): {results['mIoU']:.3f}\n\n")
        
        f.write("## Per-class IoU\n")
        for i, class_idx in enumerate(results['classes']):
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            f.write(f"- {class_name}: {results['IoU_per_class'][i]:.3f}\n")
    
    print(f"Performance metrics saved to {markdown_path}")
    
    # Compare with target metrics
    print(f"\nResults for {tunnel_id}:")
    print(f"OA {results['OA']:.3f} F1 {results['F1']:.3f} mIoU {results['mIoU']:.3f}")
    
    return results

def evaluate_instance_segmentation():
    """
    Evaluate instance segmentation performance using ring data
    """
    # Read the CSV file - using the tunnel_id parameter
    data_path = f"{input_dir}/only_label.csv"
    try:
        df = pd.read_csv(data_path)
        # Check if the required columns exist
        if 'gt_rings' not in df.columns or 'pred_rings' not in df.columns:
            print("The CSV file does not contain ring data columns (gt_rings, pred_rings)")
            return
            
        print(f"\n--- Instance Segmentation Evaluation for {tunnel_id} ---")
        print(f"Evaluating ring-based instance segmentation")
        
        # Extract ring data
        gt_rings = df['gt_rings'].values
        pred_rings = df['pred_rings'].values
        
        # Get unique ring values
        unique_gt_rings = np.unique(gt_rings)
        unique_pred_rings = np.unique(pred_rings)
        print(f"Unique ground truth rings: {unique_gt_rings}")
        print(f"Unique predicted rings: {unique_pred_rings}")
        
        # Calculate metrics for rings
        ring_accuracy = np.mean(gt_rings == pred_rings)
        print(f"Ring prediction accuracy: {ring_accuracy:.3f}")
        
        # Skip visualization for instance segmentation
        
    except Exception as e:
        print(f"Error in instance segmentation evaluation: {e}")

def main():
    """
    Main function to evaluate CSV data
    """
    print(f"=== Segmentation Evaluation Tool for Tunnel {tunnel_id} (7-segment version) ===")
    print("\nEvaluating semantic segmentation...")
    csv_results = evaluate_csv_data()
    
    # Skip instance segmentation evaluation to focus only on class metrics
    # print("\nEvaluating instance segmentation...")
    # evaluate_instance_segmentation()
    
    print(f"\n=== Evaluation Complete for Tunnel {tunnel_id} ===")

if __name__ == "__main__":
    main()

