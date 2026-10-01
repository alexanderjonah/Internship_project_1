# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization

## Project objective
Build a decision-intelligence prototype that predicts historical order lead time and evaluates factory reassignment scenarios while considering historical profitability.

## Architecture
1. Data cleaning and validation
2. Feature engineering and EDA
3. Linear Regression, Random Forest, Gradient Boosting
4. MAE, RMSE, R² comparison
5. Best-model selection
6. Factory what-if scenario analysis
7. Gross-margin context
8. Streamlit dashboard

## Critical limitation
The supplied dataset does **not** contain:
- historical factory assignment
- transportation cost
- transportation distance
- factory production capacity

Therefore, the project does not claim to learn a causal factory effect. The factory module is a transparent scenario-analysis layer using the supplied factory coordinates and a conservative distance proxy. This should be described clearly in any presentation.

## Run locally

```bash
pip install -r requirements.txt
python models/train_model.py
streamlit run app/app.py
```

The dashboard should open in a browser.

## Project structure

```text
Nassau_Candy_Project/
├── data/
│   └── Nassau_Candy.csv
├── notebooks/
│   └── Nassau_Candy_Analysis.ipynb
├── models/
│   ├── train_model.py
│   ├── lead_time_model.joblib   # generated after training
│   └── model_comparison.csv     # generated after training
├── app/
│   └── app.py
├── report/
├── requirements.txt
└── README.md
```

## Interpretation
The system is a decision-support prototype, not a production logistics optimizer. Before operational deployment, the business should provide historical factory assignments, actual shipment origin/destination, freight cost, production capacity, and service-level constraints.
