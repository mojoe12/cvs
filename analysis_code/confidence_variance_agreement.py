import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
df = pd.read_csv('merged_video.csv')

# Step 2: Create new columns
df['confidence'] = df['confidence_rater1'] + df['confidence_rater2'] + df['confidence_rater3']

# Compute mean columns
df['mean_c1'] = (df['c1_rater1'] + df['c1_rater2'] + df['c1_rater3']) / 3.0
df['mean_c2'] = (df['c2_rater1'] + df['c2_rater2'] + df['c2_rater3']) / 3.0
df['mean_c3'] = (df['c3_rater1'] + df['c3_rater2'] + df['c3_rater3']) / 3.0

# Compute variance columns (not dividing by 3 to match your formula)
df['variance_c1'] = (
    (df['c1_rater1'] - df['mean_c1']) ** 2 +
    (df['c1_rater2'] - df['mean_c1']) ** 2 +
    (df['c1_rater3'] - df['mean_c1']) ** 2
)

df['variance_c2'] = (
    (df['c2_rater1'] - df['mean_c2']) ** 2 +
    (df['c2_rater2'] - df['mean_c2']) ** 2 +
    (df['c2_rater3'] - df['mean_c2']) ** 2
)

df['variance_c3'] = (
    (df['c3_rater1'] - df['mean_c3']) ** 2 +
    (df['c3_rater2'] - df['mean_c3']) ** 2 +
    (df['c3_rater3'] - df['mean_c3']) ** 2
)

print(df.head(3))

# Step 3: Compute correlations
corr_c1 = df['variance_c1'].corr(df['confidence'])
corr_c2 = df['variance_c2'].corr(df['confidence'])
corr_c3 = df['variance_c3'].corr(df['confidence'])

# Display the results
print("Correlation between variance_c1 and confidence:", corr_c1)
print("Correlation between variance_c2 and confidence:", corr_c2)
print("Correlation between variance_c3 and confidence:", corr_c3)

# Plotting
plt.figure(figsize=(15, 4))

# Plot 1: confidence vs variance_c1
plt.subplot(1, 3, 1)
plt.scatter(df['confidence'], df['variance_c1'], alpha=0.7)
plt.title(f'confidence vs variance_c1\nr = {corr_c1:.2f}')
plt.xlabel('confidence')
plt.ylabel('variance_c1')

# Plot 2: confidence vs variance_c2
plt.subplot(1, 3, 2)
plt.scatter(df['confidence'], df['variance_c2'], alpha=0.7, color='orange')
plt.title(f'confidence vs variance_c2\nr = {corr_c2:.2f}')
plt.xlabel('confidence')
plt.ylabel('variance_c2')

# Plot 3: confidence vs variance_c3
plt.subplot(1, 3, 3)
plt.scatter(df['confidence'], df['variance_c3'], alpha=0.7, color='green')
plt.title(f'confidence vs variance_c3\nr = {corr_c3:.2f}')
plt.xlabel('confidence')
plt.ylabel('variance_c3')

plt.tight_layout()
plt.show()

