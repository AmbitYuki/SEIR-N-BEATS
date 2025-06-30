# SEIR-N-BEATS: Hierarchical Temporal Pattern Learning for Epidemic Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

## Overview

SEIR-N-BEATS is a novel hybrid framework that combines epidemiological domain knowledge with advanced neural architectures for accurate epidemic forecasting. The system integrates a modified SEIR (Susceptible-Exposed-Infectious-Recovered) model with Neural Basis Expansion Analysis for Time Series (N-BEATS) to capture both epidemiological dynamics and complex temporal patterns in epidemic data.

## Key Features

- **Hybrid Architecture**: Combines traditional epidemiological modeling with deep learning
- **Dynamic Parameter Estimation**: Time-varying infection rate modeling using logistic functions
- **Policy-Aware Adaptation**: Incorporates intervention strategies and policy changes
- **Interpretable Components**: Decomposition into trend and seasonal patterns
- **Robust Performance**: Superior noise tolerance and cross-regional generalization
- **State-of-the-Art Results**: 19.1% MAPE reduction over traditional methods

## Architecture

The framework operates in three phases:
1. **Dynamic SEIR Modeling**: Generates initial predictions with epidemiological priors
2. **Neural Pattern Learning**: Captures residual dynamics through N-BEATS
3. **Ensemble Prediction**: Combines both components for final forecasts

## Project Structure

```
SEIR-N-BEATS/
├── data/                   # Epidemic datasets and preprocessing utilities
├── examples/               # Usage examples and demonstration scripts  
├── models/                 # Core model implementations
│   ├── seir.py            # Dynamic SEIR model with time-varying parameters
│   └── nbeats.py          # N-BEATS neural architecture
├── networks/              # Neural network components and utilities
├── utils/                 # Helper functions and evaluation metrics
├── deepar.py              # Alternative deep learning baseline
├── open weight file.py    # Model checkpoint utilities
└── setup.py              # Package installation script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AmbitYuki/SEIR-N-BEATS.git
cd SEIR-N-BEATS

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

## Quick Start

```python
from models.seir_nbeats import SEIRNBEATS
from utils.data_loader import load_epidemic_data

# Load epidemic data
data = load_epidemic_data('path/to/dataset.csv')

# Initialize model
model = SEIRNBEATS(
    forecast_length=7,
    backcast_length=14,
    stacks=2,
    blocks_per_stack=1
)

# Train the model
model.fit(data['train'])

# Generate predictions
predictions = model.predict(data['test'], horizon=7)
```

## Datasets

The framework has been validated on multiple real-world datasets:
- **Early Pandemic Dataset**: Chinese provincial data (Dec 2019 - Feb 2020)
- **Regional Datasets**: Guangdong, Beijing, Shanghai epidemic data
- **Multi-temporal**: Various outbreak periods and policy scenarios

## Performance

Our method demonstrates superior performance across multiple metrics:

| Method | MAE | RMSE | MAPE (%) | PCC |
|--------|-----|------|----------|-----|
| SEIR-N-BEATS | **576.39** | **672.22** | **1.03** | **0.994** |
| SEIR-LSTM | 583.68 | 714.17 | 1.05 | 0.970 |
| N-BEATS | 854.84 | 1067.87 | 1.55 | 0.970 |
| LSTM | 999.55 | 1291.96 | 1.84 | 0.930 |

## Key Components

### Dynamic SEIR Model
- Time-varying infection rate: β(t) = Ce^(-α(t+b))/(1+e^(-α(t+b)))²
- Policy modulation: β'(t) = β(t) · (1 - P(t))
- Parameter optimization through gradient descent

### N-BEATS Architecture
- Hierarchical block structure with residual connections
- Interpretable basis functions for trend and seasonality
- Forward and backward expansion coefficients

### Policy Adaptation
- Dynamic policy impact module: P(t) = Σwᵢ · pᵢ(t)
- Adaptive parameter adjustment: θ(t) = θ₀ + Δθ · P(t)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions and feedback, please open an issue or contact the development team.

---

**Note**: This framework is designed for research purposes and should be used in conjunction with public health expertise for real-world applications.
