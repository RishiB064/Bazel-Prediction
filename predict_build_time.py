import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Author: Rishi Bharadwaj 

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# 1. Load and Clean Dataset
df = pd.read_csv("build_metrics.csv")
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

df = df.dropna(subset=['cpu_time'])
df = df[np.isfinite(df['cpu_time']) & (df['cpu_time'] > 0)].reset_index(drop=True)

NUMERIC_COLS = [c for c in [
    'num_files', 'insertions', 'deletions', 'unique_ext',
    'src_ratio', 'test_ratio', 'docs_ratio',
    'third_party_ratio', 'commit_time'
] if c in df.columns]

# Split: 300 Train / 100 Validation / 100 Test
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Dataset: {len(df)} rows | numeric features: "
      f"{NUMERIC_COLS if NUMERIC_COLS else 'none (run: python data_collection.py --features-only)'}")

SCALE = float(np.median(df['cpu_time']))

def num_matrix(dataframe):
    if not NUMERIC_COLS:
        return np.zeros((len(dataframe), 0), dtype=np.float32)
    v = dataframe[NUMERIC_COLS].fillna(0.0).values.astype(np.float32)
    return np.log1p(np.clip(v, 0, None))

def prepare_data(dataframe):
    x = {
        "prefix": tf.convert_to_tensor(dataframe["prefix"].astype(str).values, dtype=tf.string),
        "type": tf.convert_to_tensor(dataframe["type"].astype(str).values, dtype=tf.string)
    }
    nm = num_matrix(dataframe)
    for j, c in enumerate(NUMERIC_COLS):
        x[c] = tf.convert_to_tensor(nm[:, j:j + 1])
    y = tf.convert_to_tensor(dataframe["cpu_time"].values.astype(np.float32) / SCALE)
    return x, y

train_x, train_y = prepare_data(train_df)
val_x, val_y = prepare_data(val_df)
test_x, test_y = prepare_data(test_df)

# Group-mean floor: theoretical best predictor using ONLY prefix + type
gm = train_df.groupby([train_df["prefix"].astype(str),
                       train_df["type"].astype(str)])["cpu_time"].mean()
gmean = float(train_df["cpu_time"].mean())
key = pd.MultiIndex.from_arrays([test_df["prefix"].astype(str),
                                 test_df["type"].astype(str)])
floor_pred = gm.reindex(key).fillna(gmean).values
print(f"Group-mean floor on this test split: "
      f"{root_mean_squared_error(test_df['cpu_time'].values, floor_pred):.2f} ms")

# 2. Input Categorical Layers
prefix_input = tf.keras.Input(shape=(1,), name='prefix', dtype=tf.string)
type_input = tf.keras.Input(shape=(1,), name='type', dtype=tf.string)

# 3. String Lookup Encoding
prefix_lookup = tf.keras.layers.StringLookup(output_mode='one_hot')
prefix_lookup.adapt(train_df["prefix"].astype(str).values)
type_lookup = tf.keras.layers.StringLookup(output_mode='one_hot')
type_lookup.adapt(train_df["type"].astype(str).values)

prefix_encoded = prefix_lookup(prefix_input)
type_encoded = type_lookup(type_input)

# 4. Hashed Feature Crossing
cross_layer = tf.keras.layers.HashedCrossing(num_bins=64)([prefix_input, type_input])
cross_encoded = tf.keras.layers.CategoryEncoding(num_tokens=64, output_mode='one_hot')(cross_layer)

feature_list = [prefix_encoded, type_encoded, cross_encoded]
model_inputs = {"prefix": prefix_input, "type": type_input}

# Continuous Feature Normalization
for c in NUMERIC_COLS:
    inp = tf.keras.Input(shape=(1,), name=c)
    norm = tf.keras.layers.Normalization()
    nm_train = num_matrix(train_df)
    norm.adapt(nm_train[:, NUMERIC_COLS.index(c):NUMERIC_COLS.index(c) + 1])
    feature_list.append(norm(inp))
    model_inputs[c] = inp

merged_features = tf.keras.layers.Concatenate()(feature_list)

# 5. Regression Head
if NUMERIC_COLS:
    hidden = tf.keras.layers.Dense(32, activation='relu')(merged_features)
    hidden = tf.keras.layers.Dense(16, activation='relu')(hidden)
    output = tf.keras.layers.Dense(1, activation=None)(hidden)
else:
    output = tf.keras.layers.Dense(1, activation=None)(merged_features)

model = tf.keras.Model(inputs=model_inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

# 6. Model Training
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\n--- Training Deep/Cross Neural Network ---")
model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    epochs=200,
    batch_size=16,
    verbose=1,
    callbacks=[early_stop]
)

# 7. Model Evaluation
test_predictions = model.predict(test_x).flatten() * SCALE
nn_rmse = root_mean_squared_error(test_df["cpu_time"].values, test_predictions)

# 8. Gradient Boosting Benchmark
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
C_tr = oe.fit_transform(train_df[["prefix", "type"]].astype(str))
C_te = oe.transform(test_df[["prefix", "type"]].astype(str))
X_tr = np.hstack([C_tr, num_matrix(train_df)]).astype(float)
X_te = np.hstack([C_te, num_matrix(test_df)]).astype(float)

gbm = HistGradientBoostingRegressor(
    max_iter=400,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=10,
    l2_regularization=1.0,
    categorical_features=[0, 1],
    random_state=42
)
gbm.fit(X_tr, train_df["cpu_time"].values)
gbm_rmse = root_mean_squared_error(test_df["cpu_time"].values, gbm.predict(X_te))

print("\n==========================================")
print("             EVALUATION SUMMARY           ")
print("==========================================")
print(f"Group-Mean Categorical Floor: {root_mean_squared_error(test_df['cpu_time'].values, floor_pred):,.2f} ms")
print(f"Keras Hybrid NN RMSE:        {nn_rmse:,.2f} ms")
print(f"HistGradientBoosting RMSE:   {gbm_rmse:,.2f} ms")
print("==========================================")
