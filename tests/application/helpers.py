"""提供保持手工小向量含义的 512 维生产测试样本。"""

import numpy as np


def embedding(values) -> np.ndarray:
    """把可手算的分量补零至生产维度，不改变模长和点积。"""
    vector = np.asarray(values, dtype=np.float32)
    return np.pad(vector, (0, 512 - vector.size))
