# TabPFN with Feature Importance Supervision

This repository contains the implementation and experiments for my thesis project on feature-importance learning in a TabPFN-style model.

The project investigates joint learning of tabular prediction and feature importance from synthetic tasks. Synthetic training tasks are generated from linear and structural causal model priors, which provide ground-truth feature importance for supervision. The framework supports both classification and regression.

## Project Structure

```text
.
├── src/
│   ├── data/
│   │   ├── scm_task_v2/
│   │   │   ├── scm.py                          # Scalar SCM and random structural functions
│   │   │   ├── task.py                         # SCM task construction
│   │   │   ├── observation.py                  # Feature and target observation mechanisms
│   │   │   ├── priors.py                       # Prior sampling utilities
│   │   │   ├── analysis.py                     # Prior quality analysis
│   │   │   ├── program_analysis.py             # Structural-program analysis
│   │   │   ├── generate_regression_borders.py  # Regression target borders
│   │   │   ├── visualization.py                # SCM visualization
│   │   │   └── utils.py                        # Sampling utilities
│   │   │
│   │   ├── linear_task.py        # Linear synthetic task generator
│   │   ├── synthetic_task.py     # Base interface for synthetic tasks
│   │   ├── datasets.py           # Deterministic synthetic task dataset
│   │   ├── collate.py            # Synthetic-task batching and padding
│   │   ├── collate_real_data.py  # OpenML preprocessing and evaluation batching
│   │   ├── config.py             # Linear and SCM prior configurations
│   │   └── borders_100.pt        # Regression target bucket borders
│   │
│   ├── model/
│   │   ├── attention.py          # Attention modules
│   │   ├── backbone.py           # Transformer backbone
│   │   ├── encoder.py            # Tabular input encoder
│   │   └── tabpfn.py             # Prediction model and importance head
│   │
│   └── training/
│       ├── train.py              # Synthetic-task training and checkpointing
│       ├── eval.py               # Synthetic-task validation
│       ├── metrics.py            # Prediction and feature-importance metrics
│       └── helper.py             # Batch/device and DataLoader utilities
│
├── experiments/
│   ├── training/                 # Training scripts for different experimental configurations
│   ├── evaluate/                 # Real-data evaluation
│   ├── analysis/                 # Synthetic-prior analysis
│   ├── plots/                    # Plotting scripts
│   └── config.py                 # Experiment configurations
│
├── results/
│   ├── training/                 # Training logs and checkpoints
│   ├── evaluation/               # Evaluation results
│   └── figures/                  # Generated figures
│
├── checkpoints/                  # Saved model checkpoints
├── requirements.txt
└── README.md
```
