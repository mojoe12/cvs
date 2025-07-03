import pandas as pd
import os

# === Load the input CSV ===
input_path = "analysis/combined_frame.csv"
df = pd.read_csv(input_path)

# === Construct 'image' column ===
df["image"] = df["dir_name"] + "_" + df["frame_id"].astype(str) + ".jpg"

# === Compute temp_c1, temp_c2, temp_c3 ===
df["temp_c1"] = 0.5 + (
    (df["c1_rater1"] - 0.5) +
    (df["c1_rater2"] - 0.5) +
    (df["c1_rater3"] - 0.5)
) / 3

df["temp_c2"] = 0.5 + (
    (df["c2_rater1"] - 0.5) +
    (df["c2_rater2"] - 0.5) +
    (df["c2_rater3"] - 0.5)
) / 3

df["temp_c3"] = 0.5 + (
    (df["c3_rater1"] - 0.5) +
    (df["c3_rater2"] - 0.5) +
    (df["c3_rater3"] - 0.5)
) / 3

# === Compute rounded final class values ===
df["c1"] = df["temp_c1"]
df["c2"] = df["temp_c2"]
df["c3"] = df["temp_c3"]

# === Compute confidence scores ===
df["confidence_c1"] = 1#2 * abs(0.5 - df["temp_c1"])
df["confidence_c2"] = 1#2 * abs(0.5 - df["temp_c2"])
df["confidence_c3"] = 1#2 * abs(0.5 - df["temp_c3"])

# === Compute positive class counts ===
pos_c1 = df["c1"].sum()
pos_c2 = df["c2"].sum()
pos_c3 = df["c3"].sum()

# === Weight per class ===
N = len(df)
w_c1 = df["c1"].sum() / N
w_c2 = df["c2"].sum() / N
w_c3 = df["c3"].sum() / N

# === Compute weights using class frequencies ===
df["weight_c1"] = 1#df["confidence_c1"] * (df["c1"] * (1. - w_c1) + (1 - df["c1"]) * w_c1)
df["weight_c2"] = 1#df["confidence_c2"] * (df["c2"] * (1. - w_c2) + (1 - df["c2"]) * w_c2)
df["weight_c3"] = 1#df["confidence_c3"] * (df["c3"] * (1. - w_c3) + (1 - df["c3"]) * w_c3)

# === Select and reorder columns ===
output_df = df[[
    "image", "c1", "c2", "c3", "weight_c1", "weight_c2", "weight_c3"
]]

# === Save the final CSV ===
output_path = "analysis/mlc_data_no_weights.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"✅ Saved: {output_path}")
