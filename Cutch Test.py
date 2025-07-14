import pandas as pd
import joblib

model = joblib.load("xgb_main_model.pkl")
mccutchen_2025_data = {
    'Tm': 'PIT',
    'BatAge': '38',
    'G': '84',
    'PA': '340',
    'AB': '297',
    'R': '30',
    'H': '76',
    '2B': '14',
    '3B': '0',
    'HR': '8',
    'RBI': '31',
    'SB': '1',
    'CS': '0',
    'BB': '36',
    'SO': '73',
    'BA': '.256',
    'OBP': '.337',
    'OPS': '.721',
    'SLG': '.384',
    'OPS+': '100',
    'TB': '114',
    'HBP': '2',
    'SH': '0',  
    'SF': '3',
    'IBB': '1',
}

wrapped_data = {k: [v] for k, v in mccutchen_2025_data.items()}
X_mccutchen = pd.DataFrame(wrapped_data)

# Convert numeric columns to appropriate types
numeric_cols = [
    'BatAge', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS',
    'BB', 'SO', 'OPS+', 'TB', 'HBP', 'SH', 'SF', 'IBB'
]
float_cols = ['BA', 'OBP', 'OPS', 'SLG']

for col in numeric_cols:
    X_mccutchen[col] = pd.to_numeric(X_mccutchen[col])

for col in float_cols:
    X_mccutchen[col] = X_mccutchen[col].astype(float)

X_mccutchen['Tm'] = X_mccutchen['Tm'].astype('category')

# Remove columns not in model
for col in ['SLG', 'OPS+']:
    if col in X_mccutchen.columns:
        X_mccutchen.drop(col, axis=1, inplace=True)

# Add missing columns with default values
missing_cols = {
    '#Bat': 1,
    'R/G': 0.0,
    'GDP': 0,
    'LOB': 0
}
for col, default in missing_cols.items():
    if col not in X_mccutchen.columns:
        X_mccutchen[col] = default

# Reorder columns to match model
model_features = ['Tm', '#Bat', 'BatAge', 'R/G', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'OPS', 'TB', 'GDP', 'HBP', 'SH', 'SF', 'IBB', 'LOB']
X_mccutchen = X_mccutchen[model_features]

# Ensure types
X_mccutchen['Tm'] = X_mccutchen['Tm'].astype('category')
for col in ['BatAge', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'TB', 'GDP', 'HBP', 'SH', 'SF', 'IBB', 'LOB', '#Bat']:
    X_mccutchen[col] = pd.to_numeric(X_mccutchen[col])
for col in ['BA', 'OBP', 'OPS', 'R/G']:
    X_mccutchen[col] = X_mccutchen[col].astype(float)

predicted_osplus = model.predict(X_mccutchen)
print(f"Predicted OPS+ for Andrew McCutchen in 2025: {predicted_osplus[0]}")