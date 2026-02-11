# DT-CNC-Public-Benchmark

**Fully Reproducible LSTM Benchmark for CNC Tool Wear Prediction**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.xxxx/zenodo.xxxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete code and documentation for the paper:

> *"A Reproducible LSTM Benchmark for Tool Wear Prediction Using Public CNC Datasets"*  
> Submitted to *Digital Twins and Applications*.

## 🎯 Purpose

- Provide a **clean, documented, and completely reproducible** baseline for tool wear prediction.
- Enable fair comparison of future methods on **public datasets**.
- Eliminate the "black box" of private industrial data.

## 📦 Datasets Used

| Dataset | Source | Link |
|---------|--------|------|
| CNC Mill Tool Wear | Kaggle / UCI | [Download](https://www.kaggle.com/datasets/vinayak123tyagi/cnc-mill-tool-wear) |
| CNC Turning Tool Wear | Kaggle | [Download](https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear) |

**No permission or registration is required.** The datasets are directly downloadable from Kaggle.

## 🧪 Reproduce All Results

Run the following commands in a Python 3.9+ environment:

```bash
git clone https://github.com/985211-MAX/dt-cnc-public-benchmark
cd dt-cnc-public-benchmark
pip install -r requirements.txt
python run_experiments.py
