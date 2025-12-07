# Structural Long-Timeseries Learning

## Abstract
Traditional Transformers are "greedy": they optimize for the next token (+1$), often overfitting to local noise and missing long-term structural phase transitions. 

This repository implements a **Non-Greedy Structural Learning** approach. By separating "momentum" (short-term) from "structure" (long-term) using orthogonal representations, we aim to predict endogenous regime shifts (financial crises) in non-stationary time series.

## Setup & Usage (uv)
This project uses `uv` for high-performance dependency management.

```bash
# Install dependencies
uv sync

# Run training pipeline
uv run python src/train.py
```
