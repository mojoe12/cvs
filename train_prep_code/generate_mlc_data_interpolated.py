import pandas as pd
import os

# === Load the input CSV ===
input_path = "analysis/train_frame_interpolated.csv"
df = pd.read_csv(input_path)

# === Construct 'image' column ===
df["image"] = df["dir_name"].astype(str) + "_" + df["frame_id"].astype(str) + ".jpg"

# === Compute temp_c1, temp_c2, temp_c3 ===
df["c1"] = 0.5 + (
    (df["c1_rater1"] - 0.5) * df["c1_rater1_confidence_time"] +
    (df["c1_rater2"] - 0.5) * df["c1_rater2_confidence_time"] +
    (df["c1_rater3"] - 0.5) * df["c1_rater3_confidence_time"]
) / (df["c1_rater1_confidence_time"] + df["c1_rater2_confidence_time"] + df["c1_rater3_confidence_time"])

df["c2"] = 0.5 + (
    (df["c2_rater1"] - 0.5) * df["c2_rater1_confidence_time"] +
    (df["c2_rater2"] - 0.5) * df["c2_rater2_confidence_time"] +
    (df["c2_rater3"] - 0.5) * df["c2_rater3_confidence_time"]
) / (df["c2_rater1_confidence_time"] + df["c2_rater2_confidence_time"] + df["c2_rater3_confidence_time"])

df["c3"] = 0.5 + (
    (df["c3_rater1"] - 0.5) * df["c3_rater1_confidence_time"] +
    (df["c3_rater2"] - 0.5) * df["c3_rater2_confidence_time"] +
    (df["c3_rater3"] - 0.5) * df["c3_rater3_confidence_time"]
) / (df["c3_rater1_confidence_time"] + df["c3_rater2_confidence_time"] + df["c3_rater3_confidence_time"])

# === Compute confidence scores ===
df["weight_c1"] = (
    (df["c1_rater1_confidence_time"] + df["c1_rater2_confidence_time"] + df["c1_rater3_confidence_time"]) / 3
)

df["weight_c2"] = (
    (df["c2_rater1_confidence_time"] + df["c2_rater2_confidence_time"] + df["c2_rater3_confidence_time"]) / 3
)

df["weight_c3"] = (
    (df["c3_rater1_confidence_time"] + df["c3_rater2_confidence_time"] + df["c3_rater3_confidence_time"]) / 3
)

# === Select and reorder columns ===
output_df = df[[
    "image", "c1", "c2", "c3", "weight_c1", "weight_c2", "weight_c3"
]]

# === Save the final CSV ===
output_path = "analysis/train_mlc_data_interpolated.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"✅ Saved: {output_path}")
