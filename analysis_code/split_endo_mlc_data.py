import os
import pandas as pd

# File paths
input_csv = 'endoscapes/all_metadata.csv'
output_dir = 'analysis'
os.makedirs(output_dir, exist_ok=True)

# Output files
train_csv = os.path.join(output_dir, 'endo_train_mlc_data.csv')
val_csv = os.path.join(output_dir, 'endo_val_mlc_data.csv')
test_csv = os.path.join(output_dir, 'endo_test_mlc_data.csv')

# Load and rename columns
df = pd.read_csv(input_csv, usecols=['vid', 'frame', 'C1', 'C2', 'C3'])
df = df.rename(columns={'C1': 'c1', 'C2': 'c2', 'C3': 'c3'})

# Create 'image' column
df['image'] = df['vid'].astype(str) + '_' + df['frame'].astype(str) + '.jpg'

# Add weight columns
df['weight_c1'] = 1
df['weight_c2'] = 1
df['weight_c3'] = 1

# Split based on 'vid'
train_df = df[df['vid'] <= 120]
val_df = df[(df['vid'] > 120) & (df['vid'] <= 161)]
test_df = df[df['vid'] > 161]

# Save to CSV
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("✅ Done! CSVs written with renamed columns and 'image':")
print(f" - Train: {train_df.shape[0]} rows → {train_csv}")
print(f" - Val:   {val_df.shape[0]} rows → {val_csv}")
print(f" - Test:  {test_df.shape[0]} rows → {test_csv}")

