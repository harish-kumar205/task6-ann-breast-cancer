# Task 6: ANN on the Breast Cancer Dataset

A reproducible Keras workflow for the Artificial Intelligence & Machine Learning Task 6 assignment.

## Contents

- `Task_6_ANN_Comparison.ipynb`: guided notebook with preprocessing, ANN training, evaluation, training curves, confusion matrices, and baseline comparison.
- `task6_ann_comparison.py`: script version of the analysis.

## Dataset and method

Uses scikit-learn’s Breast Cancer Wisconsin Diagnostic dataset (569 rows, 30 features). The target mapping is 0 = malignant and 1 = benign. A stratified 80/20 split with random state 42 is shared by all models. StandardScaler is fit on training data only.

The neural network has Dense layers of 16 ReLU units, 8 ReLU units, and 1 sigmoid output. It is compiled with Adam and binary cross-entropy, then trained for 50 epochs with batch size 16 and a 20% validation split. Logistic Regression and Random Forest baselines are evaluated on the same held-out test set.

## Run

Install dependencies:

```bash
python -m pip install tensorflow scikit-learn pandas matplotlib
```

Open the notebook and run all cells, or run the script:

```bash
python task6_ann_comparison.py
```

The run saves training curves, confusion matrices, a model comparison CSV, and the ANN classification report in the working directory.
