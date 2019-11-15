from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np


def get_logreg_output(features_df, target_df):

    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=40)
    logreg = LogisticRegression(class_weight='balanced', n_jobs=-1)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy_score = np.round(logreg.score(X_test, y_test), 2)
    class_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report)
    class_report_df.columns = class_report_df.columns.str.upper()
    class_report_df.index = class_report_df.index.str.upper()
    class_report_df= class_report_df.round(3).transpose().\
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
