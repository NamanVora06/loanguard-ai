# Reports Directory

This directory stores evaluation metrics and explainability outputs:

- `evaluation_metrics.json`: Detailed performance metrics for TabNet and baseline models on the held-out test set (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix).
- `global_feature_importance.json`: Global feature attention scores aggregated across the test set.

To regenerate these evaluation reports:
```powershell
python src/evaluate.py
```
