import numpy as np

class ReLU():
    """
    ReLU汎用クラス
    """
    
    def __init__(self):
      self.mask = None
        
        
    def forward(self, A):
        """
        フォワードにおける活性化関数の計算
        
        Parameters
        -----------
        A : 入力(活性化前)
        
        Return
        -----------
        Z : 出力(活性化後)

        """
        
        self.mask = (A <= 0)
        Z = A.copy()
        Z[self.mask] = 0
        
        return Z
    
    
    def backward(self, dZ):
        """
        バックワードにおける活性化関数の計算
        
        Parameters
        -----------
        dZ : 入力(活性化前)
        
        Return
        -----------
        dA : 出力(活性化後)

        """
        dA = dZ.copy()
        dA[self.mask] = 0
        
        return dA