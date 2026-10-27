# Models Directory

This directory stores trained model checkpoints and preprocessing pipelines:

- `preprocessor.joblib`: Serialized `TabNetDataPreprocessor` containing category mappings, imputers, and standard scalers fitted only on training data.
- `tabnet_model.zip`: Checkpoint for the trained `TabNetClassifier`, containing PyTorch neural network weights, mask dimensions, and embedding layers.
- `baseline_lr.joblib`: Trained Scikit-Learn `LogisticRegression` baseline model for comparative evaluation.

To train or retrain these models from scratch:
```powershell
python src/train.py
```
