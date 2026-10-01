
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Nassau_Candy.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]

# Clean dates
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Order Date", "Ship Date"]).copy()

# Standardize known product-name whitespace issue(s)
df["Product Name"] = (
    df["Product Name"].astype(str).str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.replace("Wonka Bar -Scrumdiddlyumptious", "Wonka Bar - Scrumdiddlyumptious", regex=False)
)

# Target
df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

# Date features
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek
df["Order_Quarter"] = df["Order Date"].dt.quarter

# Profitability feature
df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Gross Profit"] / df["Sales"], 0)

# Features deliberately exclude Row ID, Order ID, Customer ID, dates themselves,
# Ship Date (leakage), and Postal Code (treated as an identifier rather than
# a geographic feature without a verified geocoding source).
features = [
    "Ship Mode", "Country/Region", "City", "State/Province",
    "Division", "Region", "Product ID", "Product Name",
    "Sales", "Units", "Gross Profit", "Cost", "Profit_Margin",
    "Order_Year", "Order_Month", "Order_DayOfWeek", "Order_Quarter"
]
target = "Lead Time"

X = df[features].copy()
y = df[target].copy()

categorical = [
    "Ship Mode", "Country/Region", "City", "State/Province",
    "Division", "Region", "Product ID", "Product Name"
]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=350, random_state=42, n_jobs=-1, min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42
    )
}

results = []
fitted = {}

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    })
    fitted[name] = pipe

comparison = pd.DataFrame(results).sort_values(
    ["RMSE", "MAE"], ascending=[True, True]
).reset_index(drop=True)

best_name = comparison.iloc[0]["Model"]
best_pipe = fitted[best_name]

FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

artifact = {
    "pipeline": best_pipe,
    "model_name": best_name,
    "model_comparison": comparison,
    "feature_columns": features,
    "historical_data": df,
    "factories": FACTORIES,
    "product_factory": PRODUCT_FACTORY,
    "scenario_alpha": 0.15,
}
joblib.dump(artifact, MODEL_DIR / "lead_time_model.joblib")
comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

print(comparison.round(3).to_string(index=False))
print(f"\nSaved best model: {best_name}")
