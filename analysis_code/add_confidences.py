import pandas as pd

# Load the CSVs
df_frames = pd.read_csv('analysis/combined_frame.csv')
df_videos = pd.read_csv('analysis/combined_video.csv')

# Merge on a common key — update 'dir_name' to whatever key is shared (e.g., 'dir_name')
merged_df = pd.merge(
    df_frames,
    df_videos[['dir_name', 'confidence_rater1', 'confidence_rater2', 'confidence_rater3']],
    on='dir_name',  # <-- update if your actual shared key is different
    how='left'      # keeps all frame rows, adds video info
)

# Save or display
print(merged_df.head())
merged_df.to_csv('analysis/combined_frame_with_confidence.csv', index=False)

