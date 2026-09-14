# Building, Breaking and Fixing a Neural Network — Fashion-MNIST

Deep learning coursework assignment: build a feedforward neural network end to end,
deliberately push it into overfitting, then repair it using regularisation and
hyperparameter tuning, measuring the effect of every design choice.

## Final Result

- **Test accuracy:** 89.72%
- **Macro precision:** 89.83% | **Macro recall:** 89.72% | **Macro F1:** 89.75%
- **Improvement over baseline:** +5.84 percentage points (83.88% → 89.72%)
- **Final configuration:** 2 hidden layers × 512 units (ReLU), lr=0.00118 (Adam),
  dropout=0.039, trained on the full 48,000-sample training set

## How to Reproduce

1. Go to [kaggle.com](https://kaggle.com) and create a free account (phone
   verification needed only if you want GPU acceleration — this notebook runs
   fine on CPU).
2. Create a new notebook: **Create → New Notebook**.
3. Attach the dataset: click **Add Input**, search for `Fashion-MNIST`, and add
   the dataset by **zalando-research** (980+ upvotes — this is the original).
4. Upload/paste the contents of `notebook.ipynb` from this repo into your Kaggle
   notebook, or use **File → Import Notebook** if Kaggle supports direct `.ipynb`
   import.
5. Run all cells in order (**Run All**). No GPU is required, though enabling
   **GPU T4 x2** under Session Options → Accelerator will speed up Parts 5–7.
6. Total runtime on CPU: approximately 1–2 hours end to end (Part 6 and Part 7
   are the most compute-heavy sections).

## Structure

- **Part 1:** Backpropagation implemented from scratch in NumPy, verified
  against PyTorch autograd (max gradient difference ~1e-17)
- **Part 2:** Activation function comparison (sigmoid, tanh, ReLU, leaky ReLU)
- **Part 3:** Cross-entropy vs MSE loss comparison, plus a regression sub-task
- **Part 4:** Optimiser comparison (SGD, SGD+momentum, RMSProp, Adam)
- **Part 5:** Deliberately forced overfitting on a reduced dataset
- **Part 6:** Regularisation study (L2, L1, dropout, batch norm, early stopping,
  data augmentation, more training data)
- **Part 7:** Random search with 5-fold cross-validation, final model retrained
  on the full dataset and evaluated once on the held-out test set

## Requirements

Runs entirely within Kaggle's default Python environment  no additional
installs needed (numpy, torch, matplotlib, scikit-learn, seaborn all
pre-installed).
