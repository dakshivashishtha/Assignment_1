# Sales Prediction System Using Machine Learning

A Machine Learning-based system that predicts monthly sales for a product sub-category
within a market, built on the Global Superstore dataset (2011-2014). Developed for
MDCORA Technologies Pvt. Ltd.

**Best model:** Gradient Boosting Regressor (ratio target) — R² = 0.787, RMSE ≈ 1,987,
a 37.5% error reduction versus a naive "last month repeats" baseline. See
`Sales_Prediction_Documentation.docx` for the full write-up.

---

## Folder contents

| File | What it is |
|---|---|
| `Sales_Prediction_Project.ipynb` | Full workflow: preprocessing → EDA → feature engineering → model training → evaluation |
| `sales_1.csv` | Source dataset |
| `sales_prediction_model.joblib` | Trained model, saved by the notebook's Phase 7 cell |
| `app.py` | Streamlit app — interactive prediction interface |
| `final_prediction_results.csv` | Model's predictions on the 2014 test set, actual vs. predicted |
| `Sales_Prediction_Documentation.docx` | Full project report (dataset, preprocessing, EDA, modelling, evaluation, results, limitations) |
| `Sales_Prediction_Presentation.pptx` | Summary slide deck |
| `README.md` | This file |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
jupyter
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

---

## How to run

### 1. Re-run the full analysis and re-train the model
```bash
jupyter notebook Sales_Prediction_Project.ipynb
```
Run all cells top to bottom. This regenerates every chart, retrains all six models,
prints the evaluation and cross-validation tables, and re-saves
`sales_prediction_model.joblib` and `final_prediction_results.csv`.

### 2. Use the interactive prediction app
`app.py` must be in the same folder as `sales_prediction_model.joblib`.
```bash
streamlit run app.py
```
Opens in your browser (usually `http://localhost:8501`). Pick a Sub-Category, Market,
and month, enter recent sales/price/discount, and get an instant prediction.

---

## What the model predicts

Given a **Sub-Category**, **Market**, **month**, recent sales history, average price,
and discount level, the model predicts **total monthly Sales** for that sub-category
in that market. It is a monthly demand forecast at the sub-category × market level —
not a prediction for an individual order or customer.

---

## Notes

- Data is split chronologically (train: 2011-2013, test: 2014) to reflect real
  forecasting use, not a random split.
- The target is modelled as a ratio (sales ÷ 12-month rolling average) to keep scale
  comparable across sub-categories of very different sizes, then converted back to a
  sales figure.
- Full methodology, limitations, and future improvement ideas are in
  `Sales_Prediction_Documentation.docx`.
