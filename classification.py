from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import pandas as pd
import numpy as np

def get_classify_output(features_df, target_df, active_norm):

    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.3, random_state=40)

    if active_norm == 1:
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))
        X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))
    else:
        X_train = X_train
        X_test = X_test

    random_forest = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=0,
                                            class_weight='balanced', n_jobs=-1)
    random_forest.fit(X_train, y_train)
    
    y_pred = random_forest.predict(X_test)
    accuracy_score = np.round(random_forest.score(X_test, y_test), 2)
    class_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report)
    class_report_df.columns = class_report_df.columns.str.upper()
    class_report_df.index = class_report_df.index.str.upper()
    class_report_df = class_report_df.round(3).transpose().reset_index().rename(columns={'index': ""})

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    confusion_df = pd.DataFrame(
        confusion_matrix,
        columns=sorted(target_df.unique()),
        index=sorted(target_df.unique()))
    confusion_df.index.name = 'Actual'
    confusion_df.columns.name = 'Prediction'

    confusion_df = confusion_df.stack().rename("value").reset_index()

    rf_feature_labels = features_df.columns.values.tolist()
    rf_feature_importance = random_forest.feature_importances_.tolist()

    return accuracy_score, class_report_df, confusion_df, rf_feature_labels, rf_feature_importance