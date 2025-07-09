import pandas as pd
import os

# === Load the input CSV ===
input_path = "analysis/val_frames.csv"
df = pd.read_csv(input_path)

# === Construct 'image' column ===
df["image"] = df["dir_name"] + "_" + df["frame_id"].astype(str) + ".jpg"

# === Compute c1, c2, c3 ===
df["c1"] = 0.5 + (
    (df["c1_rater1"] - 0.5) +
    (df["c1_rater2"] - 0.5) +
    (df["c1_rater3"] - 0.5)
) / 3

df["c2"] = 0.5 + (
    (df["c2_rater1"] - 0.5) +
    (df["c2_rater2"] - 0.5) +
    (df["c2_rater3"] - 0.5)
) / 3

df["c3"] = 0.5 + (
    (df["c3_rater1"] - 0.5) +
    (df["c3_rater2"] - 0.5) +
    (df["c3_rater3"] - 0.5)
) / 3

# === Compute confidence scores ===
df["weight_c1"] = 1
df["weight_c2"] = 1
df["weight_c3"] = 1

# === Select and reorder columns ===
output_df = df[[
    "image", "c1", "c2", "c3", "weight_c1", "weight_c2", "weight_c3"
]]

# === Save the final CSV ===
output_path = "analysis/val_mlc_data.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"✅ Saved: {output_path}")
