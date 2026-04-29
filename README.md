# DAT255 Project: Demand Forecasting for Inventory Restocking

This project was built for DAT255 Deep Learning Engineering. It demonstrates an
end-to-end machine learning pipeline for retail demand forecasting and inventory
restocking decisions.

The core idea is simple: use recent product demand to forecast the next 7 days,
then evaluate how different restocking policies would have performed against the
actual future demand.

## Authors

- Jakob Kallevik
- Hoang Vinh Nguyen

## Live Demo

The webapp is built as a Next.js application with a Python/PyTorch inference
runtime behind the API routes.

Live deployment:

- https://demand-forecast-demo-project.onrender.com/

Main views:

- / - interactive model demo
- /explain - plain-language guide for non-specialists
- /api/series - available product/store demand series
- /api/infer?series_id=FOODS_1_001_CA_1 - run inference for one series

## What The Demo Shows

The app lets a user:

- Select one product demand series
- Run real TCN model inference
- Inspect point forecasts and quantile forecasts
- Visualize historical demand, actual future demand and model predictions
- Compare inventory policies using decision-oriented metrics
- Read a plain-language explanation of key terms such as time series, stockout
  rate, fill rate and total cost

## Dataset

The project uses the M5 Forecasting Accuracy dataset.

Current deployment subset:

- Store: CA_1
- Category: FOODS
- Department: FOODS_1
- Number of series: 216
- Series length: 1913 daily demand values

In this project, one series means one product in one store. For example:

text
FOODS_1_001_CA_1

means product FOODS_1_001 in store CA_1.

## Model Pipeline

The final pipeline uses:

- Input window: 28 days of observed demand
- Forecast horizon: 7 days
- Point forecast TCN
- Quantile forecast TCN with quantiles 0.1, 0.5 and 0.9
- Inventory simulation with simplified costs

The quantile model is uncertainty-aware. Instead of only predicting one future
demand value, it predicts low, median and high demand scenarios.

## Decision Metrics

Forecasts are evaluated not only by forecast quality, but also by how useful
they are for inventory decisions.

Included policy metrics:

- Total cost: simulated holding penalty plus stockout penalty
- Stockout rate: share of days where inventory was too low
- Fill rate: share of demand that was fulfilled

Current demo cost assumptions:

- Holding cost: 1
- Stockout cost: 5

This means missed demand is penalized more heavily than leftover inventory.

## Repository Structure

text
project/
├── Dockerfile
├── requirements-deploy.txt
├── data/
│   └── processed/
│       └── webapp_models/
│           ├── point_tcn_5a.pt
│           └── quantile_tcn_5a.pt
├── src/
│   ├── compare_policies_5a.py
│   ├── export_webapp_series.py
│   ├── preprocessing_5a.py
│   ├── train.py
│   ├── train_quantile.py
│   ├── webapp_inference_runtime.py
│   └── models/
│       └── tcn.py
├── webapp/
│   ├── app/
│   │   ├── api/
│   │   ├── explain/
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── data/
│   │   └── ca_1_foods_1_validation_series.json
│   ├── lib/
│   └── package.json
└── results/

## Local Development

Install webapp dependencies:

npm install --prefix webapp

Run the development server:

npm run dev --prefix webapp

If port 3000 is busy:

npm run dev --prefix webapp -- -p 3001

Open:

text
http://localhost:3000
http://localhost:3000/explain
http://localhost:3000/api/series

## Docker

Build the deployment image:

docker build -t dat255-webapp .

Run locally:

docker run --rm -p 3000:3000 dat255-webapp

Test:

text
http://localhost:3000/
http://localhost:3000/explain
http://localhost:3000/api/series
http://localhost:3000/api/infer?series_id=FOODS_1_001_CA_1

If port 3000 is already allocated, either stop the old container or map the
container to another local port:

docker run --rm -p 3001:3000 dat255-webapp

Then open http://localhost:3001.

## Deployment

The app is deployed as a Docker-based web service. This is necessary because the
Next.js API route calls Python/PyTorch inference.

Important deployment files:

- Dockerfile
- .dockerignore
- requirements-deploy.txt
- src/webapp_inference_runtime.py
- data/processed/webapp_models/*.pt

The model checkpoint files are intentionally included in deployment even though
the rest of data/processed is ignored.

## Render Notes

If Render deploys but the UI does not show the newest changes, check:

1. Render is connected to the same branch you pushed.
2. The latest Render deploy shows the expected commit SHA.
3. Auto deploy is enabled, or a manual deploy was triggered after the push.
4. The browser is not showing a cached page.

The intended deployment branch is main.

## Project Status

Completed:

- Baseline forecasting and inventory simulation
- Point forecast TCN
- Quantile TCN with pinball loss
- Policy comparison on stricter 5A dataset
- Interactive webapp demo
- Plain-language explanation view
- Docker deployment setup

The project is intended as both a DAT255 final project artifact and a portfolio
demo of applied machine learning for retail decision support.