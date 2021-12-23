import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from .utils import timer


# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams["figure.figsize"] = (20, 5)
color = ["#2639E9", "#F76E6C", "#FE7715"]


class anomaly_detection:
    
    model = None
    _ratio = 0.125
    
    @classmethod
    @timer
    def train(cls, x, x_cols=None):
        if x_cols:
            anomaly_detection.model = IsolationForest(
                                            n_estimators=256
                                            , n_jobs=-1
                                            , contamination=cls._ratio
                                            , max_features=np.floor(np.sqrt(len(x_cols)))

                                        )
            cls.model.fit(x[x_cols])
        else:
            raise RuntimeError('Train Error')
        
    @classmethod
    @timer
    def predict(cls, x=None, x_cols=None):
        if x_cols:
            
            x['y'] = cls.model.predict(x[x_cols]) == -1
            x['rule'] = x[x_cols] > x[x_cols].quantile(0.5)
            x['anomaly'] = (x['y'] & x['rule']).astype(bool)
            return x[x_cols+['anomaly']]
        else:
            raise RuntimeError('Predict Error')
    
    @classmethod
    @timer
    def plot(cls, res, x_cols=None, figsize=(20, 5)):
        if x_cols:
            plt.figure(figsize=figsize, dpi=120)
            plt.plot(res.index, res[x_cols], color=color[0])
            plt.scatter(res[res['anomaly']].index, res[res['anomaly']][x_cols], color=color[1])
        else:
            raise RuntimeError('Plot Error')
    
    @classmethod
    @timer
    def execute_all(cls, x, x_cols=None, ratio = 0.125):
        cls._ratio = ratio
        if x_cols:
            cls.train(x, x_cols=x_cols)
            res = cls.predict(x, x_cols=x_cols)
            cls.plot(res, x_cols=x_cols)
            print(f"异常数据的比例为 {res['anomaly'].sum()/len(res)}")
            return res
        else:
            raise RuntimeError('Input Error')