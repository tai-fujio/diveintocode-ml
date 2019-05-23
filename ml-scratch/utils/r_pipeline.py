from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def r_pipeline(data, clf):
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'])

    # 学習
    clf.fit(X_train, y_train) 

    # 予測実施
    y_pred = clf.predict(X_test)
    
    return mean_squared_error(y_test, y_pred)