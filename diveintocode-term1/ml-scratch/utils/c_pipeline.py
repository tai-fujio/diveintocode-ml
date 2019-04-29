from sklearn.model_selection import train_test_split
from sklearn import metrics

def c_pipeline(data, clf_dict):
    accuracy_dict = {}
    confusion_matrix_dict = {}
    
    for i, clf_key in enumerate(clf_dict):
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(data[i]['X'], data[i]['y'])
        
        # 学習
        clf_dict[clf_key].fit(X_train, y_train) 

        # 予測実施
        y_pred = clf_dict[clf_key].predict(X_test)

        
        accuracy_dict[clf_key] = metrics.accuracy_score(y_test, y_pred)
        
        confusion_matrix_dict[clf_key] = metrics.confusion_matrix(y_test, y_pred)
    
    return accuracy_dict, confusion_matrix_dict