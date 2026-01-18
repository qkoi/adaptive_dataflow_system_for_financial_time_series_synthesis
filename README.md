# **`README.md`**

# History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.10143-b31b1b.svg)](https://arxiv.org/abs/2601.10143)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2601.10143)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Discipline](https://img.shields.io/badge/Discipline-Quantitative%20Finance%20%7C%20Deep%20Learning-00529B)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Data Sources](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20Binance-lightgrey)](https://finance.yahoo.com/)
[![Core Method](https://img.shields.io/badge/Method-Bi--Level%20Optimization-orange)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Analysis](https://img.shields.io/badge/Analysis-Cointegration--Aware%20Mixup-red)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Validation](https://img.shields.io/badge/Validation-Stylized%20Facts%20Fidelity-green)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Robustness](https://img.shields.io/badge/Robustness-Distributional%20Drift%20Metrics-yellow)](https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-blue.svg)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis"** by:

*   **Haochong Xia** (Nanyang Technological University)
*   **Yao Long Teng** (Nanyang Technological University)
*   **Regan Tan** (Nanyang Technological University)
*   **Molei Qin** (Nanyang Technological University)
*   **Xinrun Wang** (Singapore Management University)
*   **Bo An** (Nanyang Technological University)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and rigorous validation of financial time-series data to the training of adaptive planners and task models via bi-level optimization, culminating in the evaluation of model robustness against concept drift and non-stationarity.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_pipeline_orchestrator`](#key-callable-run_pipeline_orchestrator)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Xia et al. (2026). The core of this repository is the iPython Notebook `adaptive_dataflow_system_for_financial_time_series_synthesis_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline addresses the critical challenge of **concept drift** in financial markets by treating data augmentation not as a static preprocessing step, but as a dynamic, learnable policy.

The paper argues that models trained on static historical data fail to generalize because market dynamics evolve ($P_t(X, Y) \neq P_{t+k}(X, Y)$). This codebase operationalizes the proposed solution: a **Drift-Aware Adaptive Dataflow System** that:
-   **Validates** financial data integrity using strict K-line consistency checks ($L_t \le \min(O_t, C_t) \le \max(O_t, C_t) \le H_t$).
-   **Synthesizes** realistic financial scenarios using a parameterized manipulation module that respects cointegration relationships.
-   **Optimizes** augmentation strategies in real-time using a meta-learning Planner trained via bi-level optimization.
-   **Evaluates** robustness using rigorous distributional distance metrics (PSI, K-S, MMD) and financial stylized facts.

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Deep Learning, and Meta-Learning.

**1. Parameterized Data Manipulation Module ($\mathcal{M}$):**
A controllable synthesis engine that transforms input data while preserving economic validity.
-   **Single-Stock Transformations:** Jittering, Scaling, Magnitude Warping, Permutation, and STL Decomposition.
-   **Multi-Stock Mix-up:** Blends assets based on **Cointegration** strength. If manipulation strength $\lambda \le 0.5$, it mixes highly cointegrated pairs (fidelity); if $\lambda > 0.5$, it mixes weakly correlated pairs (stress testing).
-   **Interpolation Compensation:** Uses **Mutual Information (MI)** to ensure augmented samples retain semantic meaning.

**2. Bi-Level Optimization:**
The system learns the optimal augmentation policy by solving a nested optimization problem:
-   **Inner Loop:** The Task Model ($f_\theta$) minimizes training loss on *augmented* data.
-   **Outer Loop:** The Planner ($g_\phi$) minimizes the Task Model's loss on *real validation* data by adjusting the augmentation policy ($p, \lambda$).
$$ \min_{\phi} \mathcal{L}_{val}(f_\theta, x_{valid}) \quad \text{s.t.} \quad \theta = \arg\min_{\theta} \mathcal{L}_{train}(f_\theta, \tilde{x}_{train}) $$

**3. Adaptive Curriculum Learning:**
An overfitting-aware scheduler dynamically adjusts the proportion of data to be augmented ($\alpha$) based on the model's learning progress, implementing a soft curriculum that ramps up difficulty as the model improves.

**4. Reinforcement Learning Transfer:**
The augmentation policy learned on forecasting tasks is transferred to RL agents (DQN, PPO) to improve their robustness in trading environments with transaction costs and regime shifts.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis/blob/main/adaptive_dataflow_system_for_financial_time_series_synthesis_ipo_five.png" alt="Adaptive Dataflow System Summary" width="100%">
</div>

## Features

The provided iPython Notebook (`adaptive_dataflow_system_for_financial_time_series_synthesis_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 36 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (architectures, learning rates, augmentation settings) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, K-line consistency, and temporal alignment.
-   **Deterministic Execution:** Enforces reproducibility through seed control, deterministic sorting, and rigorous logging of all stochastic outputs.
-   **Comprehensive Evaluation:** Computes forecasting metrics (MSE, MAE), trading metrics (Sharpe Ratio, Total Return), and distributional drift metrics (PSI, K-S, MMD).
-   **Reproducible Artifacts:** Generates structured `RunContext` objects, serializable outputs, and cryptographic manifests for every intermediate result.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-4):** Ingests raw OHLCV data, validates schemas, enforces K-line constraints, and cleanses missing values.
2.  **Configuration Resolution (Task 5):** Resolves missing parameters with ground-truth defaults and hashes the configuration for provenance.
3.  **Feature Engineering (Tasks 6-10):** Computes forecasting targets, constructs sliding windows, aligns tensors for mix-up, creates chronological splits, normalizes data, and computes cointegration matrices.
4.  **Data Manipulation Module (Tasks 11-15):** Implements single-stock transformations, curation layers, multi-stock mix-up operations (CutMix, LinearMix, AmplitudeMix), target sampling (Algorithm 1), and binary mix compensation (Algorithm 2).
5.  **Adaptive Control (Tasks 16-26):** Implements the curriculum scheduler (Algorithm 3), the joint training scheme (Algorithm 4), the modular task model interface, specific architectures (GRU, LSTM, TCN, Transformer, DLinear), the Planner network, risk-aware loss, and bi-level optimization updates.
6.  **Training Pipeline (Task 27):** Orchestrates the end-to-end training of forecasting models using the adaptive planner.
7.  **RL Transfer (Tasks 28-31):** Implements the trading environment, DQN/PPO agents, and the transfer learning experiment using the pre-trained planner.
8.  **Evaluation (Tasks 32-35):** Computes trading metrics, distribution shift metrics, stylized facts fidelity, and generates t-SNE visualizations.
9.  **Orchestration (Task 36):** Unifies all components into a single `run_pipeline_orchestrator` function.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 36 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_pipeline_orchestrator`

The project is designed around a single, top-level user-facing interface function:

-   **`run_pipeline_orchestrator`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, cleansing, modeling, transfer learning, and evaluation modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `torch`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis.git
    cd adaptive_dataflow_system_for_financial_time_series_synthesis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy torch scipy statsmodels scikit-learn matplotlib pyyaml
    ```

## Input Data Structure

The pipeline requires a primary DataFrame `df_raw` with a MultiIndex `(date, ticker)` and the following columns:
1.  **`Open`**: Float, $>0$.
2.  **`High`**: Float, $\ge \max(Open, Close)$.
3.  **`Low`**: Float, $\le \min(Open, Close)$.
4.  **`Close`**: Float, $>0$.
5.  **`Volume`**: Int/Float, $\ge 0$.
6.  **`AdjClose`**: Float, $>0$ (Required for US Stocks).
7.  **`technical_indicators`**: Numeric columns as specified in the config.

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `run_pipeline_orchestrator` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    with open("config.yaml", "r") as f:
        study_config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    df_raw = generate_synthetic_financial_data()

    # 3. Execute the entire replication study.
    run_context = run_pipeline_orchestrator(
        df_raw=df_raw,
        universe="US_Stocks_Daily",
        study_config=study_config,
        output_dir="./experiment_artifacts"
    )
    
    # 4. Access results
    if run_context.training_results:
        print(run_context.training_results["LSTM"]["metrics"])
```

## Output Structure

The pipeline returns a `RunContext` object containing:
-   **`config`**: The resolved configuration dictionary.
-   **`df_clean`**: The cleansed and curated DataFrame.
-   **`tensor_data`**: Dictionary of windowed and aligned tensors.
-   **`training_results`**: Dictionary containing metrics, history, and state dicts for all trained models.
-   **`rl_results`**: Results from the RL transfer experiment.
-   **`drift_metrics`**: Dictionary of PSI, K-S, and MMD scores.
-   **`stylized_facts`**: Dictionary of fidelity metrics (ACF, Leverage Effect).
-   **`drift_plots`**: Paths to generated t-SNE plots.

## Project Structure

```
adaptive_dataflow_system_for_financial_time_series_synthesis/
│
├── adaptive_dataflow_system_for_financial_time_series_synthesis_draft.ipynb   # Main implementation notebook
├── config.yaml                                                                # Master configuration file
├── requirements.txt                                                           # Python package dependencies
│
├── LICENSE                                                                    # MIT Project License File
└── README.md                                                                  # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Global Settings:** `lookback_window`, `split_ratios`, `rolling_protocol`.
-   **Model Architectures:** `hidden_dim`, `num_layers`, `dropout` for GRU, LSTM, TCN, Transformer, DLinear.
-   **Planner Settings:** `input_dim`, `sharpe_loss_gamma`, `update_freq`.
-   **Augmentation:** `operations` list, `cointegration_threshold_lambda`.
-   **RL Environment:** `transaction_cost`, `initial_capital`, `policy_lr`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Additional Task Models:** Integrating state-of-the-art architectures like N-BEATS or TFT.
-   **Real-Time Adaptation:** Extending the pipeline to support online learning with streaming data.
-   **Multi-Asset RL:** Expanding the RL environment to support portfolio optimization across multiple assets.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{xia2026history,
  title={History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis},
  author={Xia, Haochong and Teng, Yao Long and Tan, Regan and Qin, Molei and Wang, Xinrun and An, Bo},
  journal={arXiv preprint arXiv:2601.10143},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). Adaptive Dataflow System for Financial Time-Series Synthesis: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/adaptive_dataflow_system_for_financial_time_series_synthesis
```

## Acknowledgments

-   Credit to **Haochong Xia, Yao Long Teng, Regan Tan, Molei Qin, Xinrun Wang, and Bo An** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, PyTorch, SciPy, Statsmodels, and Scikit-Learn**.

--

*This README was generated based on the structure and content of the `adaptive_dataflow_system_for_financial_time_series_synthesis_draft.ipynb` notebook and follows best practices for research software documentation.*
