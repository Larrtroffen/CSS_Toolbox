import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, r2_score
from sklearn.datasets import load_iris, load_boston, make_classification, make_regression
import matplotlib.pyplot as plt

# Helper function to create a diverse dataset for demonstration
def create_xgb_dataset(n_samples=200, n_features=10, n_informative=5, n_classes=2, task='classification'):
    """Creates a dataset, potentially with NaNs for XGBoost to handle."""
    if task == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                   n_informative=n_informative, n_classes=n_classes, 
                                   random_state=42, n_redundant=1, n_repeated=0)
    else: # regression
        X, y = make_regression(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, random_state=42)

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    # Introduce some NaNs for XGBoost to demonstrate its handling
    if n_samples > 0 and n_features > 0:
        nan_mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95]) # 5% NaNs
        df = df.mask(nan_mask)
        print(f"Introduced NaNs into {task} dataset. Total NaNs: {df.isnull().sum().sum()}")

    print(f"Created XGBoost {task} dataset: {df.shape[0]} samples, {df.shape[1]} features.")
    return df, y

# --- 1. XGBoost for Classification ---
def demonstrate_xgboost_classification(X_orig, y_orig):
    print("\n--- 1. XGBoost for Classification --- ")
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.25, random_state=42, stratify=y_orig if len(np.unique(y_orig)) > 1 else None)

    # XGBoost DMatrix (efficient data structure for XGBoost)
    # XGBoost can handle NaNs natively if enable_categorical=False (default for numeric only)
    # For mixed types with enable_categorical=True, special care is needed or prior encoding.
    # Here, assuming X_orig might contain NaNs but is otherwise numeric or XGBoost can infer types.
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan) # Explicitly tell DMatrix how NaNs are represented
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)
    print("DMatrix created for training and testing.")

    # Parameters
    params = {
        'objective': 'binary:logistic' if len(np.unique(y_orig)) == 2 else 'multi:softmax',
        'eval_metric': 'logloss' if len(np.unique(y_orig)) == 2 else 'mlogloss',
        'eta': 0.1, # learning rate
        'max_depth': 3,
        'seed': 42,
        'nthread': -1 # Use all available threads
    }
    if len(np.unique(y_orig)) > 2:
        params['num_class'] = len(np.unique(y_orig))

    # Training with early stopping
    print("Training XGBoost classifier with early stopping...")
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    num_boost_round = 100 # Max rounds
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=10, # Stop if no improvement for 10 rounds
        verbose_eval=False # Set to True or a number (e.g., 10) to see training progress
    )
    print(f"Training complete. Best iteration: {bst.best_iteration}")

    # Prediction
    y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
    if len(np.unique(y_orig)) == 2:
        y_pred = (y_pred_proba > 0.5).astype(int)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    else: # Multiclass
        y_pred = y_pred_proba.argmax(axis=1) if params['objective'] == 'multi:softprob' else y_pred_proba
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # For multi-class ROC AUC, need predict_proba and one-vs-rest or one-vs-one approach
        # y_pred_proba_mc = bst.predict(dtest, output_margin=False, iteration_range=(0, bst.best_iteration))
        # print(f"ROC AUC (OvR): {roc_auc_score(y_test, y_pred_proba_mc, multi_class='ovr'):.4f}") # Requires multi:softprob

    # Feature Importance
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(bst, ax=ax, max_num_features=10, importance_type='gain')
    plt.title("XGBoost Feature Importance (Classification)")
    # plt.show()
    plt.savefig("xgboost_classification_importance.png")
    print("Feature importance plot saved as xgboost_classification_importance.png")
    plt.close(fig)

    # Cross-validation with xgb.cv
    print("\nPerforming XGBoost CV (simplified)...")
    # Note: For xgb.cv, data needs to be in DMatrix. We already have params.
    # Remove eval_metric if it's a list for cv, or ensure it's a single string if using multiple metrics
    cv_params = params.copy()
    if isinstance(cv_params.get('eval_metric'), list):
        cv_params['eval_metric'] = cv_params['eval_metric'][0] # Take the first one for simplicity
    elif cv_params.get('objective') == 'multi:softmax' and cv_params.get('eval_metric') == 'mlogloss':
        pass # mlogloss is fine for multi:softmax
    elif cv_params.get('objective') == 'binary:logistic' and cv_params.get('eval_metric') == 'logloss':
        pass
    else: # Default to logloss/mlogloss if eval_metric is not set or incompatible
        cv_params['eval_metric'] = 'logloss' if len(np.unique(y_orig)) == 2 else 'mlogloss'
    
    try:
        cv_results = xgb.cv(
            cv_params, 
            dtrain, 
            num_boost_round=50, # Reduced rounds for CV speed
            nfold=3, 
            stratified= (len(np.unique(y_orig)) > 1), # Stratified if classification and more than 1 class
            early_stopping_rounds=5, 
            seed=42, 
            verbose_eval=False
        )
        print("CV results (mean of last round's eval metric):")
        metric_key = f'test-{cv_params["eval_metric"]}-mean'
        print(f"  {cv_params['eval_metric']} (mean): {cv_results[metric_key].iloc[-1]:.4f}")
    except Exception as e:
        print(f"xgb.cv failed: {e}. This can happen with certain parameter combinations or data issues.")

    return bst

# --- 2. XGBoost for Regression ---
def demonstrate_xgboost_regression(X_orig, y_orig):
    print("\n--- 2. XGBoost for Regression --- ")
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.25, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 3,
        'seed': 42,
        'nthread': -1
    }

    print("Training XGBoost regressor with early stopping...")
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    num_boost_round = 100
    bst_reg = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    print(f"Regression training complete. Best iteration: {bst_reg.best_iteration}")

    y_pred = bst_reg.predict(dtest, iteration_range=(0, bst_reg.best_iteration))
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(bst_reg, ax=ax, max_num_features=10, importance_type='weight') # 'weight' is another type
    plt.title("XGBoost Feature Importance (Regression)")
    # plt.show()
    plt.savefig("xgboost_regression_importance.png")
    print("Feature importance plot saved as xgboost_regression_importance.png")
    plt.close(fig)
    return bst_reg

# --- 3. XGBoost with Scikit-Learn Wrapper and GridSearchCV ---
def demonstrate_xgb_sklearn_gridsearch(X_orig, y_orig, task='classification'):
    print(f"\n--- 3. XGBoost Scikit-Learn Wrapper & GridSearchCV ({task}) --- ")
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.25, random_state=42, stratify=y_orig if task == 'classification' and len(np.unique(y_orig)) > 1 else None)

    if task == 'classification':
        # XGBoost handles NaNs when using its DMatrix, but sklearn wrappers might need pre-imputation
        # For simplicity, we'll impute here. In a real scenario, use a pipeline.
        if X_train.isnull().sum().sum() > 0: X_train = X_train.fillna(X_train.mean())
        if X_test.isnull().sum().sum() > 0: X_test = X_test.fillna(X_test.mean())
        
        model = xgb.XGBClassifier(
            objective='binary:logistic' if len(np.unique(y_orig)) == 2 else 'multi:softmax',
            eval_metric='logloss' if len(np.unique(y_orig)) == 2 else 'mlogloss',
            use_label_encoder=False, # Recommended to set to False for recent XGBoost versions
            random_state=42,
            n_estimators=50 # Keep low for speed
        )
        param_grid = {
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1],
            # 'n_estimators': [30, 50] # Already set in model, can add here too
        }
        scoring_metric = 'roc_auc' if len(np.unique(y_orig)) == 2 else 'accuracy'
    else: # Regression
        if X_train.isnull().sum().sum() > 0: X_train = X_train.fillna(X_train.mean())
        if X_test.isnull().sum().sum() > 0: X_test = X_test.fillna(X_test.mean())

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            n_estimators=50
        )
        param_grid = {
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1],
        }
        scoring_metric = 'r2'

    print(f"Performing GridSearchCV for {task} model (simplified)...")
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring=scoring_metric, verbose=0, n_jobs=-1)
    try:
        grid_search.fit(X_train, y_train) # XGBoost sklearn API can handle DataFrames directly
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best {scoring_metric} score: {grid_search.best_score_:.4f}")
        
        # Evaluate best model from grid search
        best_model = grid_search.best_estimator_
        y_pred_gs = best_model.predict(X_test)
        if task == 'classification':
            print(f"Test Accuracy (best model): {accuracy_score(y_test, y_pred_gs):.4f}")
            if len(np.unique(y_orig)) == 2:
                y_pred_proba_gs = best_model.predict_proba(X_test)[:,1]
                print(f"Test ROC AUC (best model): {roc_auc_score(y_test, y_pred_proba_gs):.4f}")
        else:
            print(f"Test R2 Score (best model): {r2_score(y_test, y_pred_gs):.4f}")
            print(f"Test RMSE (best model): {np.sqrt(mean_squared_error(y_test, y_pred_gs)):.4f}")
            
    except Exception as e:
        print(f"GridSearchCV with XGBoost wrapper failed: {e}")
        print("This might be due to data issues (e.g., all-NaN columns after split) or parameter incompatibilities.")

if __name__ == '__main__':
    print("===== XGBoost Examples Demo =====")

    print("\n*** XGBoost Classification Demo ***")
    df_class_xgb, y_class_xgb = create_xgb_dataset(task='classification')
    # XGBoost DMatrix handles NaNs, so we pass df_class_xgb directly
    bst_classifier = demonstrate_xgboost_classification(df_class_xgb, y_class_xgb)
    demonstrate_xgb_sklearn_gridsearch(df_class_xgb.copy(), y_class_xgb, task='classification') # copy to avoid modification by fillna

    print("\n\n*** XGBoost Regression Demo ***")
    df_reg_xgb, y_reg_xgb = create_xgb_dataset(task='regression')
    bst_regressor = demonstrate_xgboost_regression(df_reg_xgb, y_reg_xgb)
    demonstrate_xgb_sklearn_gridsearch(df_reg_xgb.copy(), y_reg_xgb, task='regression')

    print("\n===== XGBoost Demo Complete. Check for *_importance.png files. =====") 