from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import pandas as pd
import numpy as np


def get_logreg_output(features_df, target_df, active_norm):
    non_num_features = [col for col, dt in features_df.dtypes.items() if dt == object]
    likely_cat = {}
    for var in features_df.columns:
                likely_cat[var] = features_df[var].nunique() <= 100
    likely_cat = [k for k, v in likely_cat.items() if v is True]
    non_num_features = list(set(non_num_features + likely_cat))

    if list(non_num_features):
        lb_results_df = pd.DataFrame(pd.get_dummies(features_df[non_num_features]))
        features_df = features_df.drop(columns=non_num_features)
        features_df = pd.concat([features_df, lb_results_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=40)

    if active_norm == 1:
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))
        X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))
    else:
        X_train = X_train
        X_test = X_test

    logreg = LogisticRegression(class_weight='balanced', n_jobs=-1, solver='lbfgs', max_iter=500)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy_score = np.round(logreg.score(X_test, y_test), 2)
    class_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report)
    class_report_df.columns = class_report_df.columns.str.upper()
    class_report_df.index = class_report_df.index.str.upper()
    class_report_df = class_report_df.round(3).transpose().\
        reset_index().rename(columns={'index': ""})

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    confusion_df = pd.DataFrame(
        confusion_matrix,
        columns=sorted(target_df.unique()),
        index=sorted(target_df.unique()))
    confusion_df.index.name = 'Actual'
    confusion_df.columns.name = 'Prediction'

    confusion_df = confusion_df.stack().rename("value").reset_index()
    logit_roc_auc = np.round(metrics.roc_auc_score(y_test, logreg.predict(X_test)),3)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])

    return accuracy_score, class_report_df, confusion_df, logit_roc_auc, fpr, tpr, thresholds
