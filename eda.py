import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')

# Basic info
print("Shape:", df.shape)
print("\nClass distribution:")
print(df['Class'].value_counts())
print("\nFraud percentage:", round(df['Class'].mean() * 100, 4), "%")
print("\nMissing values:", df.isnull().sum().sum())
print("\nBasic stats:")
print(df[['Amount', 'Time']].describe())

# Plot 1 - Class imbalance
plt.figure(figsize=(6,4))
df['Class'].value_counts().plot(kind='bar', color=['steelblue','crimson'])
plt.title('Class Distribution (0=Normal, 1=Fraud)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# Plot 2 - Amount distribution by class
plt.figure(figsize=(8,4))
df[df['Class']==0]['Amount'].clip(upper=500).hist(bins=50, alpha=0.6, label='Normal', color='steelblue')
df[df['Class']==1]['Amount'].clip(upper=500).hist(bins=50, alpha=0.6, label='Fraud', color='crimson')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount (clipped at 500)')
plt.tight_layout()
plt.savefig('amount_distribution.png')
plt.show()

# Plot 3 - Correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()



# Top features correlated with fraud
correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
print("\nTop 10 features correlated with fraud:")
print(correlations.head(10))

# Plot
plt.figure(figsize=(8,5))
correlations.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Features Correlated with Class (Fraud)')
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.show()

print("\nDone! Check the 3 saved images.")