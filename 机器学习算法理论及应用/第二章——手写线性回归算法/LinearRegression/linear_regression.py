import numpy as np
from util.features import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """：
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵

        data:数据
        polynomial_degree: 是否做额外变换
        sinusoid_degree: 是否做额外变换
        normalize_data: 是否标准化数据

        :return
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training.prepare_for_training(data, polynomial_degree, sinusoid_degree,
                                                                         normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
