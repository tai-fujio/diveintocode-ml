import random
import numpy as np

def train_test_split(X, y, train_size=0.8):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    
    
    # 初期値設定
    #random_state = random_state if type(random_state) is int else  random.randint(0, 200)
    # random.shuffle(X)
    # random.shuffle(y)
    
    # 各配列をシャッフル
    np.random.shuffle(X)
    np.random.shuffle(y)
    
    # 各配列のサイズを計算
    len_X = int(train_size * len(X))
    len_y = int(train_size * len(y))
    
    # 返却用データ作成
    X_train = X[:len_X, :]
    y_train = y[:len_y, :]

    X_test = X[len_X:, :]
    y_test = y[len_y:, :]
    
    return X_train, X_test, y_train, y_test