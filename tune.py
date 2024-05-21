import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

scaled_data = pd.read_csv('scaled_data_for_training.csv')
X = scaled_data.drop(columns=['fraud_reported'])
y = scaled_data['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 600],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

xgb = XGBClassifier(objective='binary:logistic', n_jobs=4)

folds = 10

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

grid_search = GridSearchCV(xgb, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train), verbose=3)

start_time = timer()
grid_search.fit(X_train, y_train)
timer(start_time)

print('\n All results:')
print(grid_search.cv_results_)
print('\n Best estimator:')
print(grid_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search:' % (folds))
print(grid_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(grid_search.best_params_)

results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('xgb-grid-search-results-01.csv', index=False)

y_test_pred_proba = grid_search.predict_proba(X_test)
results_df = pd.DataFrame(data={'id': np.arange(len(X_test)), 'target': y_test_pred_proba[:, 1]})
results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)

