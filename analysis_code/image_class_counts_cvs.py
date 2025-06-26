import pandas as pd

# Load the CSV files
df_counts = pd.read_csv('image_class_counts.csv')
df_combined = pd.read_csv('combined_frame.csv')

# Merge them on 'frame_id' and 'dir_name'
merged_df = pd.merge(df_counts, df_combined, on=['frame_id', 'dir_name'], how='inner')

# Display the merged dataframe
print(merged_df.head())

# Optionally, save the result to a new CSV
merged_df.to_csv('merged_output.csv', index=False)

