import pandas as pd
import numpy as np

# Load the original CSV
df = pd.read_csv('analysis/combined_frame.csv')

# Sort just in case
df = df.sort_values(by=['dir_name', 'frame_id'])

# Columns to interpolate
rating_columns = [col for col in df.columns if col.startswith('c') and col != 'dir_name']

frame_rate_before = 150 # this is a magic number

# Function to interpolate per group
def interpolate_group(group):
    # Set frame_id as index for interpolation
    group = group.set_index('frame_id')
    
    # Create a new index from min to max in steps of 30
    new_index = range(group.index.min(), group.index.max() + frame_rate_before, 30) # TODO this is a debatable decision
    
    # Reindex and interpolate linearly
    group_interp = group.reindex(new_index).interpolate(method='linear')
    
    # Add frame_id back as a column
    group_interp['frame_id'] = group_interp.index
    group_interp['dir_name'] = group['dir_name'].iloc[0]
    
    # Calculate confidence as distance to the nearest original frame
    group_interp['time_distance'] = group_interp.index.to_series().apply(
        lambda x: min(abs(x - i) for i in group.index)
    )

    # Optionally add uncertainty per rating column
    for col in rating_columns:
        group_interp[f'{col}_confidence_time'] = (frame_rate_before - group_interp['time_distance']) / frame_rate_before
        group_interp[f'{col}_confidence_time'] = group_interp[f'{col}_confidence_time'].mask(abs(group_interp[f'{col}'] - 0.5) > 0.49, 1)
    
    return group_interp.reset_index(drop=True)

# Apply interpolation per dir_name group
df_interp = df.groupby('dir_name', group_keys=False).apply(interpolate_group)

# Save to new CSV
df_interp.to_csv('analysis/combined_frame_interpolated.csv', index=False)

print("Interpolated data with uncertainty saved.")

