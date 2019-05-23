import numpy as np

class Softmax():
    """
    ソフトマックス汎用クラス
    """
        
    def forward(self, X):
        """
        ソフトマックス関数で予測する
        
        Parameters
        -----------
        X : 入力(活性化前)
        
        Return
        -----------
        Z : 出力(活性化後)
        """
        
        max_X = np.max(X)
        exp_X = np.exp(X - max_X)
        sum_exp_X = np.sum(exp_X, axis=1).reshape(-1, 1)
        
        Z = exp_X / sum_exp_X
        
        return Z
        
    
    def backward(self, Z, Y):
        """
        ソフトマックス関数で予測・交差エントロピーで誤差の算出をする
        
        Parameters
        -----------
        Z : 予測値
        Y : 正解値
        
        Return
        -----------
        dA : 出力
        """
        
        dA = Z - Y
        
        # 誤差を算出
        loss = - np.sum(Y * np.log(Z), axis=1)
        
        return dA, loss