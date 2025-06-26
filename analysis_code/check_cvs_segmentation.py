import pandas as pd

# Load the CSV file
df = pd.read_csv('image_class_counts_cvs.csv')

# Ensure the columns are boolean or convert them if necessary
# This assumes that the relevant columns are either already 0/1 or True/False
# If they are strings like "True"/"False", you may need to convert them explicitly
# Create derived columns for the comparisons
df['cystic_structures'] = df['cystic_artery'] & df['cystic_duct']

# Cross-tabulations
matrix_c1 = pd.crosstab(df['c1_rater1'], df['cystic_structures'], rownames=['c1_rater1'], colnames=['cystic_artery & cystic_duct'])
matrix_c2 = pd.crosstab(df['c2_rater1'], df['calot_triangle'], rownames=['c2_rater1'], colnames=['calot_triangle'])
matrix_c3 = pd.crosstab(df['c3_rater1'], df['cystic_plate'], rownames=['c3_rater1'], colnames=['cystic_plate'])

print("Matrix for c1_rater1 vs cystic_artery & cystic_duct:")
print(matrix_c1)
print("\nMatrix for c2_rater1 vs calot_triangle:")
print(matrix_c2)
print("\nMatrix for c3_rater1 vs cystic_plate:")
print(matrix_c3)

# Cross-tabulations
matrix_c1 = pd.crosstab(df['c1_rater2'], df['cystic_structures'], rownames=['c1_rater2'], colnames=['cystic_artery & cystic_duct'])
matrix_c2 = pd.crosstab(df['c2_rater2'], df['calot_triangle'], rownames=['c2_rater2'], colnames=['calot_triangle'])
matrix_c3 = pd.crosstab(df['c3_rater2'], df['cystic_plate'], rownames=['c3_rater2'], colnames=['cystic_plate'])

print("Matrix for c1_rater2 vs cystic_artery & cystic_duct:")
print(matrix_c1)
print("\nMatrix for c2_rater2 vs calot_triangle:")
print(matrix_c2)
print("\nMatrix for c3_rater2 vs cystic_plate:")
print(matrix_c3)

# Cross-tabulations
matrix_c1 = pd.crosstab(df['c1_rater3'], df['cystic_structures'], rownames=['c1_rater3'], colnames=['cystic_artery & cystic_duct'])
matrix_c2 = pd.crosstab(df['c2_rater3'], df['calot_triangle'], rownames=['c2_rater3'], colnames=['calot_triangle'])
matrix_c3 = pd.crosstab(df['c3_rater3'], df['cystic_plate'], rownames=['c3_rater3'], colnames=['cystic_plate'])

print("Matrix for c1_rater3 vs cystic_artery & cystic_duct:")
print(matrix_c1)
print("\nMatrix for c2_rater3 vs calot_triangle:")
print(matrix_c2)
print("\nMatrix for c3_rater3 vs cystic_plate:")
print(matrix_c3)

# Find rows where the mismatch occurs
mismatch = df[(df['c1_rater1'] == 0) & (df['c1_rater2'] == 0) & (df['c1_rater3'] == 0) & (df['cystic_structures'] == 1)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c1_rater* are all 0 but cystic artery and duct are both segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c2_rater1'] == 0) & (df['c2_rater2'] == 0) & (df['c2_rater3'] == 0) & (df['calot_triangle'] == 1)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c2_rater* are all 0 but calot triangle is segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c3_rater1'] == 0) & (df['c3_rater2'] == 0) & (df['c3_rater3'] == 0) & (df['cystic_plate'] == 1)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c3_rater* are all 0 but cystic plate is segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c1_rater1'] == 1) & (df['c1_rater2'] == 1) & (df['c1_rater3'] == 1) & (df['cystic_structures'] == 0)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c1_rater* are all 1 but either cystic artery or duct is not segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c2_rater1'] == 1) & (df['c2_rater2'] == 1) & (df['c2_rater3'] == 1) & (df['calot_triangle'] == 0)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c2_rater* are all 1 but calot triangle is not segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c3_rater1'] == 1) & (df['c3_rater2'] == 1) & (df['c3_rater3'] == 1) & (df['cystic_plate'] == 0)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c3_rater* are all 1 but cystic plate is not segmented: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c2_rater1'] == 1) & (df['c2_rater2'] == 1) & (df['c2_rater3'] == 1) & (df['calot_triangle'] > 1)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c2_rater* are all 1 but calot triangle is segmented into >1 piece: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")

# Find rows where the mismatch occurs
mismatch = df[(df['c3_rater1'] == 1) & (df['c3_rater2'] == 1) & (df['c3_rater3'] == 1) & (df['cystic_plate'] > 1)]

# Display the first such row (if it exists)
if not mismatch.empty:
    print(f"First time c3_rater* are all 1 but cystic plate is segmented into >1 piece: {mismatch.iloc[0].dir_name}_{mismatch.iloc[0].frame_id}.jpg")
