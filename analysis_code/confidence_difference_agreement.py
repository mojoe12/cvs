import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
df = pd.read_csv('analysis/combined_video.csv')

df['error_c1_r1'] = abs(df['c1_rater1'] - 0.5 * df['c1_rater2'] - 0.5 * df['c1_rater3'])
corr_c1_r1 = df['error_c1_r1'].corr(df['confidence_rater1'])
df['error_c2_r1'] = abs(df['c2_rater1'] - 0.5 * df['c2_rater2'] - 0.5 * df['c2_rater3'])
corr_c2_r1 = df['error_c2_r1'].corr(df['confidence_rater1'])
df['error_c3_r1'] = abs(df['c3_rater1'] - 0.5 * df['c3_rater2'] - 0.5 * df['c3_rater3'])
corr_c3_r1 = df['error_c3_r1'].corr(df['confidence_rater1'])
df['error_c1_r2'] = abs(df['c1_rater2'] - 0.5 * df['c1_rater1'] - 0.5 * df['c1_rater3'])
corr_c1_r2 = df['error_c1_r2'].corr(df['confidence_rater2'])
df['error_c2_r2'] = abs(df['c2_rater2'] - 0.5 * df['c2_rater1'] - 0.5 * df['c2_rater3'])
corr_c2_r2 = df['error_c2_r2'].corr(df['confidence_rater2'])
df['error_c3_r2'] = abs(df['c3_rater2'] - 0.5 * df['c3_rater1'] - 0.5 * df['c3_rater3'])
corr_c3_r2 = df['error_c3_r2'].corr(df['confidence_rater2'])
df['error_c1_r3'] = abs(df['c1_rater3'] - 0.5 * df['c1_rater2'] - 0.5 * df['c1_rater1'])
corr_c1_r3 = df['error_c1_r3'].corr(df['confidence_rater3'])
df['error_c2_r3'] = abs(df['c2_rater3'] - 0.5 * df['c2_rater2'] - 0.5 * df['c2_rater1'])
corr_c2_r3 = df['error_c2_r3'].corr(df['confidence_rater3'])
df['error_c3_r3'] = abs(df['c3_rater3'] - 0.5 * df['c3_rater2'] - 0.5 * df['c3_rater1'])
corr_c3_r3 = df['error_c3_r3'].corr(df['confidence_rater3'])

# Display the results
print("Correlation between error_c1 and confidence:", corr_c1_r1)
print("Correlation between error_c2 and confidence:", corr_c2_r1)
print("Correlation between error_c3 and confidence:", corr_c3_r1)
print("Correlation between error_c1 and confidence:", corr_c1_r2)
print("Correlation between error_c2 and confidence:", corr_c2_r2)
print("Correlation between error_c3 and confidence:", corr_c3_r2)
print("Correlation between error_c1 and confidence:", corr_c1_r3)
print("Correlation between error_c2 and confidence:", corr_c2_r3)
print("Correlation between error_c3 and confidence:", corr_c3_r3)

# Plotting
plt.figure(figsize=(15, 4))

# Plot 1: confidence vs error_c1
plt.subplot(1, 3, 1)
plt.scatter(df['confidence_rater1'], df['error_c1_r1'], alpha=0.7)
plt.title(f'confidence vs error_c1_r1\nr = {corr_c1_r1:.2f}')
plt.xlabel('confidence_rater1')
plt.ylabel('error_c1_r1')

# Plot 2: confidence vs error_c2
plt.subplot(1, 3, 2)
plt.scatter(df['confidence_rater2'], df['error_c2_r2'], alpha=0.7, color='orange')
plt.title(f'confidence vs error_c2_r2\nr = {corr_c2_r2:.2f}')
plt.xlabel('confidence_rater2')
plt.ylabel('error_c2_r2')

# Plot 3: confidence vs error_c3
plt.subplot(1, 3, 3)
plt.scatter(df['confidence_rater3'], df['error_c3_r3'], alpha=0.7, color='green')
plt.title(f'confidence vs error_c3_r3\nr = {corr_c3_r3:.2f}')
plt.xlabel('confidence_rater3')
plt.ylabel('error_c3_r3')

plt.tight_layout()
plt.show()

