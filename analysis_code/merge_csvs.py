import os
import pandas as pd

# === Set the root directory where all the folders are ===
root_dir = "sages_cvs_challenge_2025/labels/"  # <-- change this!

# === Collect all video.csv paths ===
csv_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "video.csv" in filenames:
        csv_paths.append(os.path.join(dirpath, "video.csv"))

# === Combine all video.csv files ===
datavideos = []
for path in csv_paths:
    df = pd.read_csv(path)
    dir_name = os.path.basename(os.path.dirname(path))
    df["dir_name"] = dir_name
    datavideos.append(df)

# === Concatenate into a single Datavideo ===
combined_df = pd.concat(datavideos, ignore_index=True)

# === Save to a new CSV ===
combined_df.to_csv("combined_video.csv", index=False)

print("Combined CSV saved as combined_video.csv")

