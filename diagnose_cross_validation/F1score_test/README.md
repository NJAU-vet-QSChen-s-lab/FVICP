# F1-Score Analysis for Reviewer Response

## Purpose
This analysis addresses **Reviewer #3, Comment #2** regarding deep learning performance metrics transparency:

> "The DNN hyperparameter search is described extensively, but performance metrics (accuracy, F1-score, confusion matrices) are not clearly reported in the main text. Supplementary validation is mentioned but should be presented more transparently."

## What This Analysis Provides

### 1. Detailed Performance Metrics
- **Accuracy**: 96.44% (Batch 1) and 98.47% (Batch 2)
- **F1-scores**: Per-cell-type precision, recall, and F1-scores
- **Confusion matrices**: Detailed classification performance visualization

### 2. Cross-validation Results
- Independent biological replicate validation (HG_1 vs DG_1 and HG_2 vs DG_2)
- Consistent performance across different animal subjects
- Reproducibility assessment

### 3. Publication-Ready Outputs
- Summary tables for manuscript text
- High-quality figures for supplementary materials
- Detailed metrics for peer review transparency

## Files Generated

### Core Results
- `combined_detailed_metrics.csv` - Complete metrics for all cell types
- `overall_summary_metrics.csv` - Batch-level accuracy and F1-scores
- `manuscript_summary_table.csv` - Publication-ready summary

### Visualizations
- `b1_confusion_matrix_detailed.pdf/.png` - Batch 1 confusion matrix
- `b2_confusion_matrix_detailed.pdf/.png` - Batch 2 confusion matrix
- `f1_scores_by_celltype.pdf/.png` - F1-score comparison across cell types
- `overall_metrics_comparison.pdf/.png` - Summary metrics visualization

### Individual Batch Reports
- `b1_detailed_metrics.csv` - Batch 1 detailed results
- `b2_detailed_metrics.csv` - Batch 2 detailed results

## How to Run

### Prerequisites
- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- matplotlib
- seaborn

### Execution
```bash
cd diagnose_cross_validation/F1score_test
python run_analysis.py \
  --b1-data ../train_b1/fascia_b1_all_train.csv \
  --b2-data ../train_b2/fascia_b2_all_train.csv \
  --b1-model ./models/b1_fascia_b1_all_train_accuracy_99.68_model.pth \
  --b2-model ./models/b2_fascia_b2_all_train_accuracy_99.80_model.pth \
  --output-dir ./
```

### Expected Runtime
- ~2-5 minutes depending on hardware
- GPU acceleration recommended but not required

## Key Results for Manuscript

### For Section 3.5 (Results)
Replace the existing accuracy-only statement with:

> "Neural network cross-validation demonstrated that both G1 (G1 = DG₁/HG₁) and G2 (G2 = DG₂/HG₂) models achieved **96.44% and 98.47% discrimination accuracy** respectively in double-blind testing, **with F1-scores ranging from 0.91-0.97 across cell types** (Supplementary Figure 2, panels D-F)."

### For Supplementary Materials
- Reference the detailed confusion matrices and performance tables
- Include F1-score breakdown by cell type
- Cite reproducibility across independent biological replicates

## Model Architecture Details

### Batch 1 (96.44% accuracy)
- Architecture: 4 hidden layers, 64 nodes per layer
- Activation: ELU
- Optimizer: Adam (learning rate: 0.01)

### Batch 2 (98.47% accuracy)
- Architecture: 5 hidden layers, 64 nodes per layer
- Activation: ELU
- Optimizer: SGD (learning rate: 0.03)

## Cross-validation Strategy
- **Independent validation**: Models trained on one biological replicate, tested on another
- **No data leakage**: Complete separation between training and validation datasets
- **Reproducible results**: Consistent high performance across different animal subjects

## Statistical Significance
- High F1-scores (0.91-0.97) indicate balanced precision and recall
- Consistent performance across batches demonstrates model robustness
- Independent biological replicates provide confidence in generalizability

## Notes for Revision
This analysis directly addresses the reviewer's concern about transparency in reporting deep learning performance metrics. The detailed F1-scores, precision, and recall values for each cell type provide the quantitative validation requested for the deep learning feature discovery approach.

---

**Generated for CSBJ-D-25-01008 Revision**  
**Date**: September 25, 2024  
**Purpose**: Reviewer #3 Response Enhancement
