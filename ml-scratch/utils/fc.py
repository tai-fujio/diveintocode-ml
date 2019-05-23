import numpy as np

class FC:
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes1 : int
      前の層のノード数
    n_nodes2 : int
      後の層のノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer                                   # 最適化インスタンスを保持
        self.W = initializer.W(n_nodes1, n_nodes2)     # 重み初期化
        self.B = initializer.B(n_nodes2)                          # バイアス初期化
        self.HW = 0                                                     #AdaGrad用
        self.HB = 0                                                      #AdaGrad用
        self.bZ = None                                                       # beforeZ
        self.dA = None                                                      # deltaA

    def forward(self, bZ):
        """
        フォワード
        Parameters
        ----------
        Z : 次の形のndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力
        """
        
        # パラメータ更新で使うためインスタンス変数化
        self.bZ = bZ.copy()
        
        A = (bZ @ self.W) + self.B
        
        return A
    
    
    def backward(self, dA):
        """
        バックワード
        Parameters
        ----------
        dA : 次の形のndarray, shape (batch_size, n_nodes2)
            後ろから流れてきた勾配
        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """
        
        # パラメータ更新で使うためインスタンス変数化
        self.dA = dA
        
        dZ = (dA @ self.W.T)
        
        # self.W self.Bの更新
        self = self.optimizer.update(self)
              
        return dZ