import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Intro Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Heart Disease Health Indicators - EDA & Predictive Modeling\n"
        "An end-to-end Machine Learning project to predict heart disease or heart attacks using the CDC's BRFSS 2015 dataset.\n"
        "\n"
        "### Project Overview:\n"
        "- **Dataset**: CDC BRFSS 2015 survey responses (253,680 records, 22 columns)\n"
        "- **Target**: `HeartDiseaseorAttack` (Binary: 0 = No, 1 = Yes)\n"
        "- **Goal**: Build a classifier that identifies individuals at high risk of heart disease or attack, prioritizing **Recall** because in healthcare, missing a positive case (False Negative) is much more dangerous than a false alarm (False Positive)."
    ))
    
    # Cell 2: Imports Code
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve\n"
        "import joblib\n"
        "import os\n"
        "\n"
        "# Set style for plots\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams['figure.figsize'] = [10, 6]\n"
        "plt.rcParams['font.size'] = 12"
    ))
    
    # Cell 3: Data Loading Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Data Loading & Understanding\n"
        "Let's load the dataset and inspect its structure, dimensions, and columns."
    ))
    
    # Cell 4: Load Data Code
    cells.append(nbf.v4.new_code_cell(
        "# Load dataset\n"
        "df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')\n"
        "print(f\"Dataset shape: {df.shape}\")\n"
        "df.head()"
    ))
    
    # Cell 5: Inspect Info Code
    cells.append(nbf.v4.new_code_cell(
        "# Inspect column data types and missing values\n"
        "df.info()"
    ))
    
    # Cell 6: Check Nulls Code
    cells.append(nbf.v4.new_code_cell(
        "# Check for null values\n"
        "print(\"Null values per column:\")\n"
        "print(df.isnull().sum())"
    ))
    
    # Cell 7: Check Duplicates Code
    cells.append(nbf.v4.new_code_cell(
        "# Check for duplicate values\n"
        "duplicates = df.duplicated().sum()\n"
        "print(f\"Number of duplicate rows: {duplicates}\")\n"
        "if duplicates > 0:\n"
        "    # Drop duplicates to keep clean data\n"
        "    df = df.drop_duplicates().reset_index(drop=True)\n"
        "    print(f\"Shape after dropping duplicates: {df.shape}\")"
    ))
    
    # Cell 8: EDA Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n"
        "Let's understand the relationships between variables, check for class imbalance, and visualize key features."
    ))
    
    # Cell 9: Target Distribution Code
    cells.append(nbf.v4.new_code_cell(
        "# Target variable distribution\n"
        "target_counts = df['HeartDiseaseorAttack'].value_counts()\n"
        "target_pct = df['HeartDiseaseorAttack'].value_counts(normalize=True) * 100\n"
        "print(\"Target Variable Distribution (HeartDiseaseorAttack):\")\n"
        "for val, count, pct in zip(target_counts.index, target_counts.values, target_pct.values):\n"
        "    label = \"Heart Disease/Attack\" if val == 1.0 else \"No Heart Disease/Attack\"\n"
        "    print(f\"{label}: {count} ({pct:.2f}%)\")\n"
        "\n"
        "# Plot target distribution\n"
        "plt.figure(figsize=(6, 4))\n"
        "sns.countplot(x='HeartDiseaseorAttack', data=df, palette='Set2')\n"
        "plt.title('Distribution of Heart Disease or Attack')\n"
        "plt.xlabel('Heart Disease or Attack (0 = No, 1 = Yes)')\n"
        "plt.ylabel('Count')\n"
        "plt.savefig('target_distribution.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 10: Imbalance Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### Key Insight:\n"
        "The target variable is heavily imbalanced (~90.6% negative and ~9.4% positive). "
        "We must handle this imbalance during model training (using `class_weight='balanced'`) "
        "and focus on metrics like **Recall** and **F1-score** rather than accuracy."
    ))
    
    # Cell 11: Correlation Heatmap Code
    cells.append(nbf.v4.new_code_cell(
        "# Correlation matrix\n"
        "plt.figure(figsize=(16, 12))\n"
        "corr_matrix = df.corr()\n"
        "sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n"
        "plt.title('Correlation Matrix of All Variables')\n"
        "plt.savefig('correlation_matrix.png', bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "# Top correlated features with target\n"
        "print(\"Correlation with HeartDiseaseorAttack (sorted):\")\n"
        "print(corr_matrix['HeartDiseaseorAttack'].sort_values(ascending=False))"
    ))
    
    # Cell 12: High BP vs Heart Disease Code
    cells.append(nbf.v4.new_code_cell(
        "# Visualize relationship between HighBP and Heart Disease\n"
        "plt.figure(figsize=(8, 5))\n"
        "sns.barplot(x='HighBP', y='HeartDiseaseorAttack', data=df, palette='viridis')\n"
        "plt.title('Heart Disease Rate by High Blood Pressure Status')\n"
        "plt.xlabel('High Blood Pressure (0 = No, 1 = Yes)')\n"
        "plt.ylabel('Heart Disease Rate (Proportion)')\n"
        "plt.savefig('bp_vs_heart_disease.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 13: Gen Health vs Heart Disease Code
    cells.append(nbf.v4.new_code_cell(
        "# Visualize relationship between General Health (GenHlth) and Heart Disease\n"
        "# GenHlth: 1 = Excellent, 5 = Poor\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.barplot(x='GenHlth', y='HeartDiseaseorAttack', data=df, palette='magma')\n"
        "plt.title('Heart Disease Rate by General Health Rating')\n"
        "plt.xlabel('General Health Rating (1 = Excellent, 5 = Poor)')\n"
        "plt.ylabel('Heart Disease Rate')\n"
        "plt.savefig('health_vs_heart_disease.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 14: BMI vs Heart Disease Code
    cells.append(nbf.v4.new_code_cell(
        "# Analyze BMI distribution for Heart Disease vs No Heart Disease\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.kdeplot(data=df[df['HeartDiseaseorAttack'] == 0], x='BMI', fill=True, label='No Heart Disease', alpha=0.5)\n"
        "sns.kdeplot(data=df[df['HeartDiseaseorAttack'] == 1], x='BMI', fill=True, label='Heart Disease', alpha=0.5)\n"
        "plt.title('BMI Distribution by Heart Disease Status')\n"
        "plt.xlabel('BMI')\n"
        "plt.xlim(10, 60)\n"
        "plt.legend()\n"
        "plt.savefig('bmi_distribution.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 15: Split & Scale Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Preprocessing & Split\n"
        "Now, we split the data into training and testing sets, and standardize numerical variables (`BMI`, `MentHlth`, `PhysHlth`) using `StandardScaler`."
    ))
    
    # Cell 16: Preprocessing & Splitting Code
    cells.append(nbf.v4.new_code_cell(
        "# Define features (X) and target (y)\n"
        "X = df.drop('HeartDiseaseorAttack', axis=1)\n"
        "y = df['HeartDiseaseorAttack']\n"
        "\n"
        "# Train-Test Split (80% train, 20% test, stratify to maintain target ratio)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        "print(f\"Training set size: {X_train.shape}\")\n"
        "print(f\"Testing set size: {X_test.shape}\")\n"
        "\n"
        "# Scale numerical columns: BMI, MentHlth, PhysHlth\n"
        "num_cols = ['BMI', 'MentHlth', 'PhysHlth']\n"
        "scaler = StandardScaler()\n"
        "\n"
        "# Fit on training data and transform both training and testing sets\n"
        "X_train_scaled = X_train.copy()\n"
        "X_test_scaled = X_test.copy()\n"
        "\n"
        "X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])\n"
        "X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])\n"
        "\n"
        "# Save the scaler for use in deployment\n"
        "joblib.dump(scaler, 'scaler.joblib')\n"
        "print(\"Scaler saved as scaler.joblib\")"
    ))
    
    # Cell 17: Model Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Model Training & Hyperparameter Tuning\n"
        "We will perform hyperparameter tuning for both models to find the best configuration:\n"
        "1. **Logistic Regression** tuned using `GridSearchCV`\n"
        "2. **Random Forest Classifier** tuned using `RandomizedSearchCV` on a stratified subset of training data (for speed), then trained on the full training set."
    ))
    
    # Cell 18: Logistic Regression Code
    cells.append(nbf.v4.new_code_cell(
        "# 1. Hyperparameter Tuning for Logistic Regression\n"
        "print(\"Tuning Logistic Regression Hyperparameters using GridSearchCV...\")\n"
        "lr_param_grid = {\n"
        "    'C': [0.01, 0.1, 1.0],\n"
        "    'solver': ['liblinear', 'lbfgs']\n"
        "}\n"
        "lr_grid_search = GridSearchCV(\n"
        "    estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),\n"
        "    param_grid=lr_param_grid,\n"
        "    cv=3,\n"
        "    scoring='roc_auc',\n"
        "    n_jobs=1\n"
        ")\n"
        "lr_grid_search.fit(X_train_scaled, y_train)\n"
        "\n"
        "print(f\"Best parameters for Logistic Regression: {lr_grid_search.best_params_}\")\n"
        "lr_model = lr_grid_search.best_estimator_\n"
        "\n"
        "# Predictions\n"
        "lr_preds = lr_model.predict(X_test_scaled)\n"
        "lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]\n"
        "\n"
        "print(\"\\n--- Tuned Logistic Regression Classification Report ---\")\n"
        "print(classification_report(y_test, lr_preds))\n"
        "print(f\"ROC-AUC Score: {roc_auc_score(y_test, lr_probs):.4f}\")"
    ))
    
    # Cell 19: Random Forest Code
    cells.append(nbf.v4.new_code_cell(
        "# 2. Hyperparameter Tuning for Random Forest using RandomizedSearchCV\n"
        "# Since the dataset is large (~200,000 rows in training), we perform RandomizedSearchCV\n"
        "# on a stratified 10% subset of training data to save time, then train the final model on full training data.\n"
        "print(\"Tuning Random Forest Hyperparameters using RandomizedSearchCV...\")\n"
        "\n"
        "# Subsampling 10% of the training data\n"
        "_, X_tune, _, y_tune = train_test_split(\n"
        "    X_train_scaled, y_train, test_size=0.10, random_state=42, stratify=y_train\n"
        ")\n"
        "print(f\"Subset size for hyperparameter tuning: {X_tune.shape}\")\n"
        "\n"
        "rf_param_dist = {\n"
        "    'n_estimators': [50, 100, 150],\n"
        "    'max_depth': [8, 12, 16],\n"
        "    'min_samples_leaf': [3, 5, 7],\n"
        "    'class_weight': ['balanced']\n"
        "}\n"
        "\n"
        "rf_random_search = RandomizedSearchCV(\n"
        "    estimator=RandomForestClassifier(random_state=42, n_jobs=1),\n"
        "    param_distributions=rf_param_dist,\n"
        "    n_iter=5,\n"
        "    cv=3,\n"
        "    scoring='roc_auc',\n"
        "    random_state=42,\n"
        "    n_jobs=1\n"
        ")\n"
        "rf_random_search.fit(X_tune, y_tune)\n"
        "\n"
        "print(f\"Best parameters found: {rf_random_search.best_params_}\")\n"
        "\n"
        "# Train the final Random Forest model on the FULL training set using the best parameters\n"
        "print(\"\\nTraining final Random Forest model on the full training set with best parameters...\")\n"
        "rf_model = RandomForestClassifier(**rf_random_search.best_params_, random_state=42, n_jobs=1)\n"
        "rf_model.fit(X_train_scaled, y_train)\n"
        "\n"
        "# Predictions\n"
        "rf_preds = rf_model.predict(X_test_scaled)\n"
        "rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]\n"
        "\n"
        "print(\"\\n--- Tuned Random Forest Classification Report ---\")\n"
        "print(classification_report(y_test, rf_preds))\n"
        "print(f\"ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}\")"
    ))
    
    # Cell 20: ROC Curves Code
    cells.append(nbf.v4.new_code_cell(
        "# Plot ROC Curves\n"
        "plt.figure(figsize=(8, 6))\n"
        "lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)\n"
        "rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)\n"
        "\n"
        "plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probs):.4f})')\n"
        "plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.4f})')\n"
        "plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')\n"
        "plt.xlabel('False Positive Rate')\n"
        "plt.ylabel('True Positive Rate')\n"
        "plt.title('ROC Curve Comparison')\n"
        "plt.legend()\n"
        "plt.savefig('roc_curve_comparison.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 21: CM Code
    cells.append(nbf.v4.new_code_cell(
        "# Plot Confusion Matrix for Random Forest\n"
        "plt.figure(figsize=(6, 5))\n"
        "cm = confusion_matrix(y_test, rf_preds)\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n"
        "            xticklabels=['No Disease', 'Disease'],\n"
        "            yticklabels=['No Disease', 'Disease'])\n"
        "plt.xlabel('Predicted')\n"
        "plt.ylabel('Actual')\n"
        "plt.title('Confusion Matrix - Random Forest')\n"
        "plt.savefig('confusion_matrix_rf.png', bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 22: Feature Importance Code
    cells.append(nbf.v4.new_code_cell(
        "# Feature Importance from Random Forest\n"
        "importances = rf_model.feature_importances_\n"
        "feature_names = X.columns\n"
        "rf_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)\n"
        "\n"
        "plt.figure(figsize=(12, 8))\n"
        "sns.barplot(x=rf_importances.values, y=rf_importances.index, palette='viridis')\n"
        "plt.title('Feature Importance - Random Forest')\n"
        "plt.xlabel('Importance Score')\n"
        "plt.ylabel('Features')\n"
        "plt.savefig('feature_importance.png', bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print(\"Top 10 features sorted by importance:\")\n"
        "print(rf_importances.head(10))"
    ))
    
    # Cell 23: Serialization Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Model Serialization\n"
        "The Random Forest model performs better (higher ROC-AUC score and balanced classification report). We save this model for use in our Streamlit web application."
    ))
    
    # Cell 24: Save Model Code
    cells.append(nbf.v4.new_code_cell(
        "# Save the model\n"
        "model_filename = 'heart_disease_model.joblib'\n"
        "joblib.dump(rf_model, model_filename)\n"
        "print(f\"Model saved successfully to {model_filename}!\")"
    ))
    
    nb['cells'] = cells
    
    with open('heart_disease_eda_modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Jupyter Notebook created successfully at heart_disease_eda_modeling.ipynb")

if __name__ == '__main__':
    create_notebook()
