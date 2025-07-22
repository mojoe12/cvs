import os
import pandas as pd
import ast

# File paths
input_csv = 'endoscapes/all_metadata.csv'
output_dir = 'analysis'
os.makedirs(output_dir, exist_ok=True)

# Output files
train_csv = os.path.join(output_dir, 'endo_train_mlc_data.csv')
val_csv = os.path.join(output_dir, 'endo_val_mlc_data.csv')
test_csv = os.path.join(output_dir, 'endo_test_mlc_data.csv')

# Load and rename columns
df = pd.read_csv(input_csv, usecols=['vid', 'frame', 'is_ds_keyframe', 'C1', 'C2', 'C3', 'cvs_annotator_1', 'cvs_annotator_2', 'cvs_annotator_3'])
df = df[df['is_ds_keyframe']]
df = df.rename(columns={'C1': 'c1', 'C2': 'c2', 'C3': 'c3', 'frame': 'frame_id', 'vid' : 'dir_name'})

# Define a function to parse stringified lists
def parse_list(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [None, None, None]  # Handle empty or invalid entries

# Apply parsing and extract new columns
for i in range(1, 4):
    col = f'cvs_annotator_{i}'
    parsed = df[col].apply(parse_list)
    df[[f'c1_rater{i}', f'c2_rater{i}', f'c3_rater{i}']] = pd.DataFrame(parsed.tolist(), index=df.index)

# Optional: drop the original annotator columns if no longer needed
df.drop(columns=[f'cvs_annotator_{i}' for i in range(1, 4)], inplace=True)

# Create 'image' column
df['image'] = df['dir_name'].astype(str) + '_' + df['frame_id'].astype(str) + '.jpg'

# Add weight columns
df['weight_c1'] = 1./6.
df['weight_c2'] = 1./6.
df['weight_c3'] = 1./6.

# Split based on 'vid'
train_df = df[df['dir_name'] <= 120]
val_df = df[(df['dir_name'] > 120) & (df['dir_name'] <= 161)]
test_df = df[df['dir_name'] > 161]

# Save to CSV
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("✅ Done! CSVs written with renamed columns and 'image':")
print(f" - Train: {train_df.shape[0]} rows → {train_csv}")
print(f" - Val:   {val_df.shape[0]} rows → {val_csv}")
print(f" - Test:  {test_df.shape[0]} rows → {test_csv}")

