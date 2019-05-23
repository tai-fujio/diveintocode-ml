import numpy as np

class SGD:
    """
    確率的勾配降下法
    Parameters
    ----------
    lr : 学習率
    """
    
    def __init__(self, lr):
        self.lr = lr
        
        
    def update(self, layer):
        """
        ある層の重みやバイアスの更新
        Parameters
        ----------
        layer : 更新前の層のインスタンス

        Returns
        ----------
        layer : 更新後の層のインスタンス
        """
        
        # 重みとバイアスを更新
        layer.W -= self.lr * (layer.bZ.T @ layer.dA) / len(layer.dA)
        layer.B -= np.mean(self.lr * layer.dA, axis=0)
        
        return layer