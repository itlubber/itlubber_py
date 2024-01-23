import pandas as pd
import numpy as np

import scipy as sp
import psutil#获取内存占有率

from sklearn.model_selection import GridSearchCV#GridSearchCV（网络搜索交叉验证）

#from sklearn.cross_validation import KFold#构造交叉验证数据集
from sklearn.metrics import log_loss
from sklearn import metrics#混淆矩阵常用模块

#Random Forest
from sklearn.ensemble import RandomForestClassifier#分类
#from sklearn.ensemble import RandomForestRegressor#回归
#from sklearn.tree import DecisionTreeRegressor

#XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

#LightGBM
import lightgbm as lgb
#from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split  # 将矩阵随机划分为训练子集和测试子集,并返回划分好的训练集、测试集样本和训练集、测试集标签
from sklearn.model_selection import StratifiedKFold  # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同

#from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings("ignore")  # 把程序打包后，不显示告警信息

import time  # 处理时间戳
from itertools import product
import itertools  # 提供了非常有用的用于操作迭代对象的函数
import copy

pd.set_option('display.max_rows', None)#显示所有行
pd.set_option('display.max_columns', None)#显示所有列
'''
data_feature = pd.read_csv('C:/Users/Amber/Desktop/ad_data/data_feature.csv', sep=',')#特征工程全部结束后的数据
'''
data_feature = pd.read_csv('C:/Users/Amber/Desktop/ad_data/data_feature.csv', sep=',',engine = 'python',iterator=True)
loop = True
chunkSize = 1000
chunks = []
index=0
while loop:
    try:
        print(index)
        chunk = data_feature.get_chunk(chunkSize)
        chunks.append(chunk)
        index+=1

    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print('开始合并')
data_feature = pd.concat(chunks, ignore_index= True)
#以上代码规定用迭代器分块读取，并规定了每一块的大小，即chunkSize，这是指定每个块包含的行数。

data_feature.drop(['Unnamed: 0'],axis=1,inplace=True)#删除第一列原本生成的index。若设置参数inplace=True，则原数据发生改变
#print(data_feature.head(1))
#print(data_feature.shape) #(478087, 376)

data_feature.drop(['instance_id'
                    ,'item_id','item_category_list','item_property_list','item_brand_id','item_city_id'
				    ,'user_id'
					,'context_id','context_timestamp','context_page_id','predict_category_property'
					,'shop_id','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description'
					,'realtime'],axis=1,inplace=True)#删除不需要的特征17个 不删'is_trade'
'''
#尝试删除属性列表类所有单属性 提高模型的auc 尝试失败 auc下降
item_category_list_name = ['item_category_list'+str(i) for i in range(1,3)]
item_property_list_name = ['item_property_list'+str(i) for i in range(10)]
predict_category_property_name = ['predict_category_property'+str(i) for i in range(5)]
all_name = item_category_list_name+item_property_list_name+predict_category_property_name
data_feature.drop(all_name,axis=1,inplace=True)
'''
data_model = data_feature
#print(data_model.shape)#(478087, 359)
train = data_model[(data_model['day'] >= 18) & (data_model['day'] <= 22)]
#print(train.shape)#(357066, 359)
test = data_model[(data_model['day'] >= 23) & (data_model['day'] <= 24)]
#print(test.shape)#(121021, 359)
y = train['is_trade'].values
train.drop(['is_trade'],axis=1,inplace=True)
X = train
print(X.shape)#(357066, 358)
#print(y.shape)#(357066,)
y_test = test['is_trade'].values
test.drop(['is_trade'],axis=1,inplace=True)
X_test = test
print(X_test.shape)#(121021, 358)
#print(y_test.shape)#(121021,)
print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
###------------1、RF
#随机森林不能自动处理缺失值，因此新建特征在rf这里用0填充
#1.1 rf调参 (两次)
X.fillna('0')
y.fillna('0')
X_test.fillna('0')
y_test.fillna('0')


param_test = {'n_estimators': [50,100,200,400,600,800,1000]}
estimator = RandomForestClassifier(bootstrap=False,oob_score=False,n_jobs=1,criterion='gini')
gsearch = GridSearchCV(estimator , param_grid = param_test,scoring='roc_auc', cv=5, verbose=1)
gsearch.fit(X,y)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

param_test1 = {'max_depth': [3,5,7,9]
	           ,'min_samples_leaf': [1,50,100,200]}
estimator1 = RandomForestClassifier(bootstrap=False,oob_score=False,n_jobs=1,criterion='gini',n_estimators= )
gsearch1 = GridSearchCV(estimator1 , param_grid = param_test1,scoring='roc_auc', cv=5, verbose=1)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#1.2 rf建模
rf_model = RandomForestClassifier(bootstrap=False
                                 ,oob_score=False
							     ,n_jobs=1
                                 ,criterion='gini'
							     ,n_estimators= 200
                                 ,max_depth= 7
								 ,min_samples_leaf= 100
                                 ,verbose=False
                                 ,max_feature='sqrt'
                                 )
rf_model = rf_model.fit(X,y)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_auc = metrics.roc_auc_score(y_test, rf_pred)
print(rf_auc)
print('误差 ', log_loss(y_test, rf_pred))

'''
#1.3 rf预测---10折交叉验证
kf = KFold(len(X), n_folds=10)
    NN_auc = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index],
        Mat_Train[test_index]
        y_train, y_test = Mat_Labels[train_index],
        Mat_Labels[test_index]
        N_auc, clf = RFandom(X_train, y_train, Mat_test, target_test)
        NN_auc.append(N_auc)
        mean_auc = mean(NN_auc)
        print 'AUC均值：',mean_auc
    return mean_auc, clf
'''
###------------2、XGBoost
#2.1 xgb调参
#第一步：第一步：确定学习率和对n_estimators进行参数调优
params = {'boosting_type': 'gbdt'
          ,'objective': 'binary'
		  ,'learning_rate': 0.1
		  ,'max_depth': 5
          ,'min_child_weight': 1
		  ,'gamma': 0
		  ,'subsample': 0.8
		  ,'colsample_bytree': 0.8
		  ,'scale_pos_weight':1
		 }#'scale_pos_weight':1因为类别十分不平衡
data_train = xgb.Dataset(X, y, silent=True)
cv_results = xgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse'
                    ,early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2019)
print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])

#第二步：对max_depth 和 min_child_weight进行参数调优
params_test1={ 'max_depth':range(3,10,2)#3,5,7,9
              ,'min_child_weight':range(1,6,2)#1,3,5
xgb_model = xgb.XGBClassifier(objective='binary'
                              ,learning_rate=0.1
		                      ,gamma=0
		                      ,subsample=0.8
		                      ,colsample_bytree=0.8
		                      ,scale_pos_weight=1)
gsearch1 = GridSearchCV(estimator=xgb_model, param_grid=params_test1, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
gsearch1.fit(X,y)
means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
#第三步：对gamma进行参数调优
params_test2={ 'gamma':[0,0.1,0.2]}
xgb_model = xgb.XGBClassifier(objective='binary'
                              ,learning_rate=0.1
		                      ,max_depth=5
                              ,min_child_weight=5
		                      ,subsample=0.8
		                      ,colsample_bytree=0.8
		                      ,scale_pos_weight=1)
gsearch2 = GridSearchCV(estimator=xgb_model, param_grid=params_test1, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
gsearch2.fit(X,y)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
#第四步：对subsample 和 colsample_bytree进行参数调优
params_test3={'subsample':[i/10.0 for i in range(7,10)]#0.7,0.8,0.9
              ,'colsample_bytree':[i/10.0 for i in range(7,10)]}
xgb_model = xgb.XGBClassifier(objective='binary'
                              ,learning_rate=0.1
		                      ,max_depth=5
                              ,min_child_weight=5
							  ,gamma=0
		                      ,scale_pos_weight=1)
gsearch3 = GridSearchCV(estimator=xgb_model, param_grid=params_test1, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
gsearch3.fit(X,y)
means = gsearch3.cv_results_['mean_test_score']
params = gsearch3.cv_results_['params']
#第五步：降低学习率（对learning_rate进行参数调优）
params = {'boosting_type': 'gbdt'
          ,'objective': 'binary'
		  ,'learning_rate': 0.01
		  ,'max_depth': 5
          ,'min_child_weight':5
		  ,'gamma':0
		  ,'subsample':0.9
		  ,'colsample_bytree':0.7
		  ,'scale_pos_weight':1
		 }
data_train = xgb.Dataset(X, y, silent=True)
cv_results = xgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse'
                    ,early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2019)
print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])

#2.2 xgb建模
xgb_model = xgb.XGBClassifier(boosting_type: 'gbdt'
                              ,objective='binary'
		                      ,learning_rate=0.01
							  ,num_boost_round=262
		                      ,max_depth=5
                              ,min_child_weight=5
		                      ,gamma=0
		                      ,subsample=0.9
		                      ,colsample_bytree=0.7
							  ,scale_pos_weight=1)
xgb_model = xgb_model.fit(X,y)
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = metrics.roc_auc_score(y_test, xgb_pred)
print(lgb_auc)
print('误差 ', log_loss(y_test, xgb_pred))
###------------3、LightGBM
#3.1 lgb调参
#第一步：确定学习率和对n_estimators进行参数调优
params = {'boosting_type': 'gbdt'
          ,'objective': 'binary'
		  ,'learning_rate': 0.1
		  ,'max_depth': 6
          ,'num_leaves':50
		  ,'min_data_in_leaf':20
		  ,'min_sum_hessian_in_leaf':0.001
		  ,'bagging_fraction': 0.8
		  ,'feature_fraction' : 0.8}
data_train = lgb.Dataset(X, y, silent=True)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse'
                    ,early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
'''
best n_estimators: 23
best cv score: 0.13626336116481058
'''
#第二步：对max_depth 和 num_leaves进行参数调优
#创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目),lgb.LGBMRegressor objective = 'regression'
params_test1={'max_depth': range(5,8,1)#5,6,7
              ,'num_leaves':[30,50,80]}#30,50,80
lgb_model = lgb.LGBMClassifier(objective='binary'
                              ,learning_rate = 0.1
							  ,n_estimators = 23
							  ,min_data_in_leaf=20
		                      ,min_sum_hessian_in_leaf=0.001							  
                              ,subsample = 0.8# 随机采样比例，0.5-1 小欠拟合，大过拟合
                              ,colsample_bytree = 0.8)# 训练每棵树时用来训练的特征的比例
gsearch1 = GridSearchCV(estimator=lgb_model, param_grid=params_test1, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
#verbose:日志显示,0为不在标准输出流输出日志信息,1为输出进度条记录,2为每个epoch输出一行记录
gsearch1.fit(X,y)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
#第三步：对min_data_in_leaf 和 min_sum_hessian_in_leaf进行参数调优
params_test2={'min_data_in_leaf': [18, 20, 22]
              ,'min_sum_hessian_in_leaf':[0.001, 0.002]}
lgb_model = lgb.LGBMClassifier(objective='binary'
                              ,learning_rate=0.1
							  ,n_estimators= 23
							  ,max_depth= 5
                              ,num_leaves= 30
                              ,subsample = 0.8
                              ,colsample_bytree = 0.8)
gsearch2 = GridSearchCV(estimator=lgb_model, param_grid=params_test2, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
gsearch2.fit(X,y)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
#第四步：对feature_fraction 和 bagging_fraction进行参数调优
params_test3={'feature_fraction': [ 0.7, 0.8, 0.9]
              ,'bagging_fraction': [0.7, 0.8, 0.9]}
lgb_model = lgb.LGBMClassifier(objective='binary'
                              ,learning_rate=0.1
							  ,n_estimators=23
							  ,max_depth= 5
                              ,num_leaves= 30
							  ,min_data_in_leaf= 18
							  ,min_sum_hessian_in_leaf= )
gsearch3 = GridSearchCV(estimator=lgb_model, param_grid=params_test3, scoring='roc_auc', cv=5, verbose=1, n_jobs=1)
gsearch3.fit(X,y)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
#第五步：降低学习率（对learning_rate进行参数调优）
params = {'boosting_type': 'gbdt'
          ,'objective': 'binary'
		  ,'learning_rate': 0.01
		  ,'max_depth': 5
          ,'num_leaves': 30
		  ,'min_data_in_leaf': 18
		  ,'min_sum_hessian_in_leaf': 0.001
		  ,'bagging_fraction':0.9
		  ,'feature_fraction':0.8}
data_train = lgb.Dataset(X, y, silent=True)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse'
                    ,early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
'''
best n_estimators: 304
best cv score: 0.1373289694495718
'''
#3.2 lgb建模
lgb_model = lgb.LGBMClassifier(objective = 'binary'
                              ,learning_rate = 0.01
							  ,n_estimators = 304
							  ,max_depth = 5
                              ,num_leaves = 30
							  ,min_data_in_leaf = 18
							  ,min_sum_hessian_in_leaf = 0.001
							  ,bagging_fraction = 0.9
							  ,feature_fraction = 0.8)
lgb_model = lgb_model.fit(X,y)
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
lgb_auc = metrics.roc_auc_score(y_test, lgb_pred)
print(lgb_auc)
print('误差 ', log_loss(y_test, lgb_pred))
'''
LGBMClassifier(bagging_fraction=0.9, boosting_type='gbdt', class_weight=None,
        colsample_bytree=1.0, feature_fraction=0.8,
        importance_type='split', learning_rate=0.01, max_depth=5,
        min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=18,
        min_split_gain=0.0, min_sum_hessian_in_leaf=0.001,
        n_estimators=304, n_jobs=-1, num_leaves=30, objective='binary',
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
0.6753889979657166
误差  0.0828657938258884
'''
#3.3 lgb重要变量
import matplotlib.pylab as plt
plt.figure(figsize=(12,6))
lgb.plot_importance(lgb_model, max_num_features=10)
plt.title("Featurertances")
plt.show()

###------------4、stacking
def stacking(X_train, y_train, test):
    # 3个一级分类器
    clfs = [
            RandomForestClassifier(n_estimators=200,random_state=0,bootstrap=False,oob_score=False,n_jobs=1
                                   ,criterion='gini',max_depth= 7,min_samples_leaf= 100,verbose=False,max_feature='sqrt'),
            xgb.XGBClassifier(boosting_type: 'gbdt',objective='binary',learning_rate=0.01,num_boost_round=262,max_depth=5
			                  ,min_child_weight=5,gamma=0,subsample=0.9,colsample_bytree=0.7,scale_pos_weight=1,nthread=-1, seed=2019),
            lgb.LGBMClassifier(objective = 'binary',learning_rate = 0.01,n_estimators = 304,max_depth = 5,num_leaves = 30
							  ,min_data_in_leaf = 18,min_sum_hessian_in_leaf = 0.001,bagging_fraction = 0.9,feature_fraction = 0.8)
            ]
    # 二级分类器的X_train, test
    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test.shape[0], len(clfs)))
    # 3个分类器进行5_folds预测
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for i, clf in enumerate(clfs):
        print('training model:',i+1)
        dataset_blend_test_j = np.zeros((test.shape[0], n_folds))  # 每个分类器的单次fold预测结果
        logloss_mean=[]
        for j, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            tr_x = X_train[train_index]
            tr_y = y_train[train_index]
            te_x = X_train[test_index]
            te_y = y_train[test_index]
            print(j+1,'fold training')
            if i<3:
                clf.fit(tr_x, tr_y)
            else:
                clf.fit(tr_x, tr_y,eval_set=[(te_x, te_y)], early_stopping_rounds=1000)
            dataset_blend_train[test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
            dataset_blend_test_j[:, j] = clf.predict_proba(test)[:,1]
            print(j+1,'fold logloss:',logloss(te_y,dataset_blend_train[test_index, i]))
            logloss_mean.append(logloss(te_y,dataset_blend_train[test_index, i]))
            # print dataset_blend_train
        dataset_blend_test[:, i] = dataset_blend_test_j.mean(1)
        print('model', i + 1, 'logloss:', np.array(logloss_mean).mean())
        # 二级分类器进行预测
    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)
    clf.fit(dataset_blend_train, y_train)
    proba = clf.predict_proba(dataset_blend_test)[:, 1]
    return proba
