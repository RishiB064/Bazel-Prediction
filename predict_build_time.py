import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Author: Rishi Bharadwaj Ramesh
# Description: Linear regression model (Keras API) with feature crossing for predicting Bazel CPU-time.

# 1. Load Data
df = pd.read_csv("build_metrics.csv")
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Split into 300 Train / 100 Validation / 100 Test
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

def prepare_data(dataframe):
    # Convert string columns directly into strict TensorFlow string tensors
    x = {
        "prefix": tf.convert_to_tensor(dataframe["prefix"].astype(str).values, dtype=tf.string),
        "type": tf.convert_to_tensor(dataframe["type"].astype(str).values, dtype=tf.string)
    }
    y = tf.convert_to_tensor(dataframe["cpu_time"].values, dtype=tf.float32)
    return x, y

train_x, train_y = prepare_data(train_df)
val_x, val_y = prepare_data(val_df)
test_x, test_y = prepare_data(test_df)

# 2. Input Layers
prefix_input = tf.keras.Input(shape=(1,), name='prefix', dtype=tf.string)
type_input = tf.keras.Input(shape=(1,), name='type', dtype=tf.string)

# 3. Bucketization (String Lookup into One-Hot Encoding)
prefix_vocab = ["site/", "src/test", "src/main", "tools/", "src/tools", "bazelci/", "scripts/", "third_party/"]
type_vocab = ["JAVA", "C/C++", "python", "Starlark"]

prefix_encoded = tf.keras.layers.StringLookup(vocabulary=prefix_vocab, output_mode='one_hot')(prefix_input)
type_encoded = tf.keras.layers.StringLookup(vocabulary=type_vocab, output_mode='one_hot')(type_input)

# 4. Feature Crossing
#cross_layer = tf.keras.layers.HashedCrossing(num_bins=100)([prefix_input, type_input])
#cross_encoded = tf.keras.layers.CategoryEncoding(num_tokens=100, output_mode='one_hot')(cross_layer)

# Concatenate all features
merged_features = tf.keras.layers.Concatenate()([prefix_encoded, type_encoded])

# 5. Linear Regression Model (Single Dense Unit)
output = tf.keras.layers.Dense(1, activation=None)(merged_features)
model = tf.keras.Model(inputs=[prefix_input, type_input], outputs=output)

# Compiled with SGD Optimizer
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), 
    loss='mean_squared_error'
)

print("Training model...")
# Fit the model
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=30, batch_size=16, verbose=1)

# 6. Evaluation
test_loss = model.evaluate(test_x, test_y, verbose=0)
print(f"\nTesting Loss (MSE): {test_loss}")

test_predictions = model.predict(test_x).flatten()
rmse = root_mean_squared_error(test_y, test_predictions)
print(f"Model RMSE with Feature Crossing: {rmse:.2f} milliseconds")