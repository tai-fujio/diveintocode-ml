import numpy as np

class HeInitializer:
    """
    Heによる初期化
    """
    
    def W(self, n_nodes1, n_nodes2):
        """
        重みの初期化
        Parameters
        ----------
        n_nodes1 : int
          前の層のノード数
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        W : 次の形のndarray, shape (n_nodes1, n_nodes2)
          重さ
        """
        
        W = np.random.randn(n_nodes1, n_nodes2) / np.sqrt(n_nodes1) * np.sqrt(2)

        return W
    
    
    def B(self, n_nodes2):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B : 次の形のndarray, shape (n_nodes2,)
          バイアス
        """
        
        B = np.zeros(n_nodes2)

        return B