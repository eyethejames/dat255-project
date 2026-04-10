# DAT255 Project — Uncertainty-Aware Demand Forecasting for Inventory Restocking

## Overview

This project is part of the course **DAT255 Deep Learning Engineering**.  
The goal is to build an end-to-end deep learning pipeline for **retail demand forecasting** and connect the forecasts to **inventory restocking decisions**.

Instead of only predicting future demand, we want to investigate whether **uncertainty-aware forecasts** can improve downstream business decisions such as stock replenishment.

The project is based on the **M5 Forecasting dataset (Walmart sales)** and is currently scoped to **one store and one product department** in order to get the full pipeline working before scaling up.

---

## Project Goal

We want to answer the following question:

> Can uncertainty-aware demand forecasts reduce inventory cost and improve service level compared to simpler baseline policies?

More concretely, the project combines:

- **Time series forecasting**
- **Deep learning models**
- **Inventory simulation**
- **Decision evaluation**

This makes the project more than a standard forecasting task:  
the forecasts are used to drive actual **restocking policies**, which are then evaluated with decision-oriented metrics.

---

## Dataset

We use the **M5 Forecasting - Accuracy** dataset from Kaggle:

- https://www.kaggle.com/competitions/m5-forecasting-accuracy

For the first implementation phase we are working with:

- **Store:** `CA_1`
- **Category:** `FOODS`
- **Department:** `FOODS_1`

This subset currently contains:

- **216 product series**
- daily sales history from the M5 dataset

---

## Current Pipeline

The project pipeline is structured as follows:

1. **Preprocess data**
2. **Create supervised windows**
   - input window: 28 days
   - forecast horizon: 7 days
3. **Train forecasting model**
4. **Generate forecasts**
5. **Run inventory simulation**
6. **Compare policies**
7. **Evaluate both forecast quality and decision quality**

---

## Input Features

The initial model uses the following inputs:

- last **28 days of historical sales**
- simple calendar information (to be expanded later)
- item identifier

Current project settings:

- **Input window:** 28 days
- **Forecast horizon:** 7 days
- **Lead time:** 7 days

---

## Models

### Baselines
We start with simple baselines to establish a reference point:

1. **Point forecast baseline**
2. **Fixed safety stock baseline**

### Main model
The main deep learning model will be a:

- **Temporal Convolutional Network (TCN)**

We start with a **point forecast TCN**, and then extend it to **quantile forecasting**.

### Uncertainty-aware extension
Later in the project, the TCN will be modified to predict quantiles:

- 0.1
- 0.5
- 0.9

This will allow us to model uncertainty and connect it directly to inventory policies.

---

## Evaluation Metrics

### Forecast metrics
We currently plan to use:

- **MAE** (Mean Absolute Error)
- **Pinball Loss**
- **Coverage**

### Decision metrics
To evaluate the actual business usefulness of the forecasts, we also use:

- **Total Cost**
  - holding cost
  - stockout cost
  - order cost
- **Stockout Rate**
- **Fill Rate**

---

## Milestones

### Milestone 1 — Baseline pipeline
Goal:
- get the full end-to-end pipeline working without deep learning

Completed work:
- data loading
- subset filtering
- sliding windows
- train/validation/test split
- baseline forecasts
- MAE
- inventory simulation
- decision metrics

Status: **Completed**

---

### Milestone 2 — Point forecast with TCN
Goal:
- implement and train a first deep learning model that predicts 7 days ahead from 28 days of history

Planned work:
- convert training data to PyTorch tensors
- implement TCN in PyTorch
- train the model
- evaluate on validation/test data
- compare against baseline MAE

Status: **Completed**

---

### Milestone 3 — Quantile forecasting with TCN
Goal:
- extend the TCN from point forecasting to uncertainty-aware forecasting

Planned work:
- output quantiles (0.1, 0.5, 0.9)
- use pinball loss
- evaluate coverage
- connect quantile forecasts to inventory decisions

Status: **In progress**

---

### Milestone 4 — Policy comparison and analysis
Goal:
- compare baseline policies and uncertainty-aware policies

Planned work:
- compare cost, stockouts and fill rate
- analyze when uncertainty-aware forecasting helps most
- reflect on limitations and assumptions

Status: **Planned**

---

### Milestone 5 — Deployment / demo
Goal:
- expose the trained model and decision pipeline in a simple interactive interface

Possible features:
- choose product / series
- generate forecast
- visualize forecast interval
- recommend restocking quantity
- compare policies

Status: **Planned**

---

## Current Repository Structure

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── baselines.py
│   ├── train.py
│   └── models/
│       └── tcn.py
│
├── results/
├── configs/
└── README.md