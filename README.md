# HiClass Project

This repository contains the implementation of the **Hierarchical Text Classification** project using **HiClass** — a hierarchical classification framework.  
It integrates **three base classifiers**:
- **SGD (Stochastic Gradient Descent)**
- **RG (Random Forest)**
- **LR (Logistic Regression)**

and combines them with **three hiclass strategies** to evaluate performance under different configurations.

---

## 🧩 Repository Structure

- **`pipeline_explanation.ipynb`** → Start here!  
  This notebook walks you through the full workflow — from data preparation to model evaluation.

- **`models/`** → Contains all module implementations (base classifiers and HiClass wrappers).

- **`helper/`** → Includes all helper functions used across the project (e.g., data processing, evaluation utilities).

- **`display_units/`** → Provides the **GUI** components for visual interaction and result presentation.

---

## 🚀 Purpose

This project aims to explore how hierarchical classification can improve model interpretability and performance when combined with optimized base classifiers.  
The modular design enables easy experimentation and extension to new strategies or datasets.
