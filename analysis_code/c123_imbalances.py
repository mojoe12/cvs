import pandas as pd
from collections import Counter

# Load the CSV
df = pd.read_csv('analysis/train_mlc_data.csv')

# Round c1, c2, c3 to 0 or 1
for col in ['c1', 'c2', 'c3']:
    df[col] = df[col].round().astype(int)

# Create a tuple of (c1, c2, c3) for each row
combinations = list(zip(df['c1'], df['c2'], df['c3']))

# Count the frequency of each combination
comb_counter = Counter(combinations)

# Print counts for all 8 possible combinations
print("Label Combination Counts (c1, c2, c3):")
for c1 in [0, 1]:
    for c2 in [0, 1]:
        for c3 in [0, 1]:
            combo = (c1, c2, c3)
            count = comb_counter.get(combo, 0)
            print(f"{combo}: {count}")

