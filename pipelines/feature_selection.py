import dtale
import functools
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
#data\diabetes_dataset.csv
health_data=pd.read_csv("./data/diabetes_dataset.csv")
print(health_data.shape)

#sample change

# Generic function to plot feature ranges versus proportion of positive outcome ('yes')
def plot_feature_ranges_vs_outcome(df, features, outcome_col, bins=15):
    # Ensure only rows with valid 'yes' and 'no' in the outcome column are mapped
    df = df[df[outcome_col].isin(['yes', 'no'])].copy()



    # Map 'yes' to 1 and 'no' to 0 in the outcome column
    df[outcome_col] = df[outcome_col].map({'yes': 1, 'no': 0})

    # Drop rows where any feature or outcome is missing
    df = df.dropna(subset=features + [outcome_col])

    for feature in features:
        # Create dynamic bins for each feature using pandas.cut
        df[f'{feature}_range'] = pd.cut(df[feature], bins=bins)

        # Group by the feature range and calculate the proportion of 'yes'
        feature_vs_outcome = df.groupby(f'{feature}_range')[outcome_col].mean().reset_index()

        # Plot the feature range versus the proportion of 'yes' outcome
        plt.figure(figsize=(10, 6))
        sns.barplot(x=f'{feature}_range', y=outcome_col, data=feature_vs_outcome)
        plt.title(f'Proportion of "yes" Outcome vs {feature} Ranges')
        plt.xlabel(f'{feature} ranges')
        plt.ylabel('Proportion of "yes" Outcome')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'./dataplots/{features[0]}_to_diabetes.png')


#Perform AUTO-EDA using Dtale
# d = dtale.show(health_data)
# d

# Create a countplot for the target variable 'Outcome'
sns.countplot(x='Outcome', data=health_data)
# Set title and labels
plt.title('Count of Diabetes (Diabetes)')
plt.xlabel('Diabetes (Nn = No Diabetes, Yes = Diabetes)')
plt.ylabel('Count')

# Show plot
#plt.show()
plt.savefig('./dataplots/target_distribution.png')


# Compute the correlation matrix
df = health_data.drop(columns=['Outcome'], errors='ignore')
corr_matrix = df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))  # Set the figure size for better readability
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Set the title
plt.title('Correlation Heatmap of Heart Disease Dataset Features')

# Show the plot
#plt.show()
plt.savefig('./dataplots/feature_correlation.png')

# Drop features that have a high correlation (above a certain threshold), you can follow these steps:
# Compute the correlation matrix.
# Identify pairs of features that have a correlation coefficient greater than a specified threshold (e.g., 0.9).
# Drop one of the highly correlated features from each pair to avoid redundancy.

# Compute the correlation matrix
corr_matrix = df.corr().abs()  # Use absolute values to consider both positive and negative correlations

# Select the upper triangle of the correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.9 (or your preferred threshold)
threshold = 0.9
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

# Drop highly correlated features
df_reduced = df.drop(columns=to_drop)


# Print the remaining features
print("Remaining features after dropping highly correlated ones:", df_reduced.columns)


#relation of age to diabetes
features = ['Age']
plot_feature_ranges_vs_outcome(health_data, features, outcome_col='Outcome', bins=10)

#relation of number of pregnancies to diabetes
features = ['Pregnancies']
plot_feature_ranges_vs_outcome(health_data, features, outcome_col='Outcome', bins=5)

#BMI to diabetes
features = ['BMI']
plot_feature_ranges_vs_outcome(health_data, features, outcome_col='Outcome', bins=10)

#BP to diabetes
features = ['BloodPressure']
plot_feature_ranges_vs_outcome(health_data, features, outcome_col='Outcome', bins=10)

#Insulin to diabetes
features = ['Insulin']
plot_feature_ranges_vs_outcome(health_data, features, outcome_col='Outcome', bins=10)

