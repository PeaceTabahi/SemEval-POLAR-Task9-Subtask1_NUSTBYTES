# Multilingual Hate Speech Detection

Binary hate speech classifier using XLM-RoBERTa-Large for English, Spanish, and German text.

## Requirements

```bash
pip install torch transformers pandas numpy scikit-learn
```

## Dataset Format

CSV files with columns: `text`, `polarization` (0/1)

## Usage

Run the Jupyter notebook cells in order:
1. Setup and imports
2. Load data
3. Train model
4. Evaluate

## Model Features

- XLM-RoBERTa-Large base model
- Class-balanced focal loss
- Multi-sample dropout ensemble
- Test-time augmentation
- Optimized for Macro F1-Score

## Training

- **GPU:** T4 or P100 (Kaggle compatible)
- **Time:** 2-3 hours
- **Epochs:** 18
- **Batch Size:** 8 (effective: 32 with gradient accumulation)

## Configuration

Key parameters in the notebook:
- Learning rate: 8e-6
- Max length: 128 tokens
- Dropout: 0.2
- Focal loss: alpha=0.25, gamma=2.0

## Files

- `xlm-roberta-final-multilingual-polarization-detection.ipynb` - Main notebook
- `best_xlm_roberta_macro_f1.pt` - Trained model checkpoint
- `plots/` - Training visualizations
- `results/` - Evaluation metrics
