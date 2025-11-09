# Baseline Models - Polarization Detection

## Overview

This folder contains our baseline implementations for POLAR @ SemEval-2026 Task 9, focusing on multilingual polarization detection across English, Spanish, and German.
We implemented four baseline models of increasing complexity: TF-IDF with Logistic Regression (69.3% F1), Bidirectional LSTM (66.1% F1), BERT-base with language-specific models (74.5% F1), and XLM-RoBERTa multilingual (80.4% F1). The traditional TF-IDF approach captured basic word frequency patterns, while the Bi-LSTM added sequential understanding but struggled with class imbalance, particularly in German (59.5% F1 for polarized class). Fine-tuning BERT separately for each language showed significant improvement through transfer learning, and XLM-RoBERTa achieved the best performance with its cross-lingual capabilities, reaching 81.8% F1 on Spanish.
All models generate predictions, confusion matrices, training curves, and performance metrics in the results/ and plots/ folders, using Macro F1-score to fairly evaluate both polarized and non-polarized classes.

