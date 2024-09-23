""""
H2O AutoML is an advanced AutoML framework designed to automate the process of building 
and optimizing machine learning models. By conducting a comprehensive search across multiple 
algorithms (such as Gradient Boosting Machines, Random Forests, XGBoost, and more), 
H2O AutoML automatically selects the best model based on performance metrics such as AUC, 
RMSE, or log loss. It simplifies the model development process by performing hyperparameter tuning, 
ensembling, and model stacking, allowing users to obtain high-performing models with 
minimal manual intervention.

1. H2O AutoML supports both classification and regression tasks and integrates with popular 
languages like Python and R. One of the key features of H2O AutoML is its leaderboard, 
which ranks the models according to their performance, allowing users to choose the best model for
their specific use case. Additionally, H2O provides flexibility for exporting models in various formats (such as MOJO and binary models) for easy deployment in production.

2. H2O AutoML is highly scalable, making it suitable for handling large datasets and complex models. 
Its ease of use and powerful optimization capabilities make it a valuable tool for both data scientists 
and business users who want to streamline their machine learning workflows without sacrificing model
performance.

3. Supports Multiple Algorithms: Automatically trains and tunes a variety of models including GBM,
 XGBoost, Random Forest, GLM, and Deep Learning. Automatic Model Ensembling: 
 Combines the best-performing models to create a more robust, stacked ensemble for improved performance.

4. Scalability: Suitable for large datasets and distributed environments, making it ideal for 
big data applications. Model Interpretability: Can be paired with XAI tools like SHAP and LIME to provide transparent insights into the model's decision-making process. Exportability: Models can be exported in MOJO or binary formats for fast, production-ready scoring without requiring a full H2O cluster.

"""
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
df = pd.read_csv('./data/diabetes_dataset.csv')

# Assuming 'Outcome' is the target column
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize h2o
h2o.init()

# Convert pandas dataframe to H2O Frame
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# Define target and features
target = 'Outcome'
features = X_train.columns.tolist()

# Run AutoML
aml = H2OAutoML(max_models=20, seed=42, verbosity='info')
aml.train(x=features, y=target, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Evaluate on the test set
preds = aml.leader.predict(test)
print(preds.head())



# Define a directory to save the models
save_dir = "./models"  # Replace with your desired directory

# Save each model by iterating over the list of model IDs

# Retrieve all model IDs from the leaderboard
model_ids = lb['model_id'].as_data_frame().iloc[:, 0].tolist()

print(model_ids)  # List of model IDs



model_paths = {}
for model_id in model_ids:
    model = h2o.get_model(model_id)  # Get the model object
    model_path = h2o.save_model(model=model, path=save_dir, force=True)
    model_paths[model_id] = model_path  # Store the path for future reference

print("All models have been saved.")

best_model = aml.leader

best_model_save_dir = "./predictions"  # Replace with your desired directory

# Save the best model
best_model_path = h2o.save_model(model=best_model, path=best_model_save_dir, force=True)

# Print the path where the best model is saved
print(f"Best model saved to: {best_model_path}")

new_model_name = 'best_diabetes_prediction'
new_model_path = os.path.join(best_model_save_dir, new_model_name)

# Rename the model directory (if needed)
os.rename(best_model_path, new_model_path)

# Confirm the model is saved with the new name
print(f"Best model saved and renamed to: {new_model_path}")


#XAI - explainability
import matplotlib.pyplot as plt

# Run model.explain() and save each plot
best_model.explain(test)

# Save the current figure as an image
# The figure must be saved before the next one is generated
xsave_dir = './xplainable_ai'
# For each type of plot that model.explain() generates, save the plot
plt.figure()  # Ensures you're capturing the current plot
plt.savefig(os.path.join(xsave_dir, "variable_importance.png"))
print("Saved Variable Importance plot.")

# Partial Dependence Plot (if applicable)
plt.figure()
plt.savefig(os.path.join(save_dir, "partial_dependence_plot.png"))
print("Saved Partial Dependence plot.")

# SHAP Summary Plot (if applicable)
plt.figure()
plt.savefig(os.path.join(save_dir, "shap_summary.png"))
print("Saved SHAP Summary plot.")
