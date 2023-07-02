#!/usr/bin/env python
# coding: utf-8

# # 2022 MathorCup 大数据 IssueB 复赛

# # 语音业务数据分析

# ## 导入第三方库

# In[1]:


import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
import warnings
warnings.filterwarnings("ignore")


# ## 读取经过剔除数据的附件1与附件3

# In[2]:


dataOne=pd.read_csv("语音业务Sample.csv",encoding='gbk')
dataThree=pd.read_excel("附件3语音业务用户满意度预测数据.xlsx",sheet_name='语音')


# In[3]:


dataOne


# In[4]:


dataThree


# ## 数据预处理

# ### 数据标准化

# In[5]:


StandardTransform = dataOne[['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                             '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                             '终端品牌','当月ARPU','当月MOU',
                             '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                             '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']]
StandardTransformScaler = sp.StandardScaler()
StandardTransformScaler = StandardTransformScaler.fit(StandardTransform)
StandardTransform = StandardTransformScaler.transform(StandardTransform)
StandardTransform = pd.DataFrame(StandardTransform)
StandardTransform.columns = ['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                             '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                             '终端品牌','当月ARPU','当月MOU',
                             '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                             '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']
StandardTransform


# In[6]:


dataOneLeave=dataOne.loc[:,~dataOne.columns.isin(['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                                                  '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                                                  '终端品牌','当月ARPU','当月MOU',
                                                  '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                                                  '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计'])]


# In[7]:


dataOneNewStandard=pd.concat([dataOneLeave, StandardTransform],axis=1)
dataOneNewStandard


# In[8]:


dataOneNewStandard.columns=['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度','语音通话稳定性',
                            '是否遇到过网络问题','居民小区','办公室','高校',
                            '商业街','地铁','农村','高铁',
                            '其他，请注明','手机没有信号','有信号无法拨通','通话过程中突然中断',
                            '通话中有杂音、听不清、断断续续','串线','通话过程中一方听不见','其他，请注明.1',
                            '是否投诉','是否关怀用户','是否4G网络客户（本地剔除物联网）','外省语音占比',
                            '外省流量占比','是否5G网络客户','脱网次数','mos质差次数',
                            '未接通掉话次数','4\\5G用户','套外流量（MB）','套外流量费（元）',
                            '语音通话-时长（分钟）','省际漫游-时长（分钟）','终端品牌',
                            '当月ARPU','当月MOU','前3月ARPU','前3月MOU',
                            'GPRS总流量（KB）','GPRS-国内漫游-流量（KB）','客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']
dataOneNewStandard


# ## 机器学习

# ### "语音通话整体满意度"学习

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[10]:


XdataOneFirst=dataOneNewStandard.loc[:,~dataOneNewStandard.columns.isin(['语音通话整体满意度','网络覆盖与信号强度',
                                                                         '语音通话清晰度','语音通话稳定性'])]
ydataOneFirst=dataOneNewStandard['语音通话整体满意度']
XdataOneFirst_train, XdataOneFirst_test, ydataOneFirst_train, ydataOneFirst_test = train_test_split(XdataOneFirst, ydataOneFirst, test_size=0.2, random_state=2022)


# #### 决策树，随机森林

# In[11]:


DecisionTreeFirst = DecisionTreeClassifier(random_state=2022)
RandomForestFirst = RandomForestClassifier(random_state=2022)
DecisionTreeFirst = DecisionTreeFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
RandomForestFirst = RandomForestFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
RandomForestFirst_score = RandomForestFirst.score(XdataOneFirst_test, ydataOneFirst_test)
RandomForestFirst_score


# #### XGBoost

# In[12]:


from xgboost import XGBClassifier

XGBFirst = XGBClassifier(learning_rate=0.01,
                         n_estimators=14,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0.,
                         subsample=1,
                         colsample_btree=1,
                         scale_pos_weight=1,
                         random_state=2022,
                         slient=0)
XGBFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
XGBFirst_score = XGBFirst.score(XdataOneFirst_test, ydataOneFirst_test)
XGBFirst_score


# #### KNN

# In[13]:


from sklearn.neighbors import KNeighborsClassifier

KNNFirst = KNeighborsClassifier()
KNNFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
KNNFirst_score = KNNFirst.score(XdataOneFirst_test, ydataOneFirst_test)
KNNFirst_score


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights':['uniform'],
                          'n_neighbors':[k for k in range(2,20)]},
                         {'weights':['distance'],
                          'n_neighbors':[k for k in range(2,20)],
                          'p':[p for p in range(1,5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid = KNN_turing_param_grid,
                                      n_jobs = -1,
                                      verbose = 2)
KNN_turing_grid_search.fit(XdataOneFirst_train, ydataOneFirst_train)


# In[15]:


KNN_turing_grid_search.best_score_


# In[16]:


KNN_turing_grid_search.best_params_


# In[17]:


KNNFirst_new = KNeighborsClassifier(n_neighbors=25, p=2, weights='distance')
KNNFirst_new.fit(XdataOneFirst_train, ydataOneFirst_train)
KNNFirst_new_score = KNNFirst_new.score(XdataOneFirst_test, ydataOneFirst_test)
KNNFirst_new_score


# #### 支持向量机

# In[18]:


from sklearn.svm import SVC

SVMFirst = SVC(random_state=2022)
SVMFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
SVMFirst_score = SVMFirst.score(XdataOneFirst_test, ydataOneFirst_test)
SVMFirst_score


# #### lightgbm

# In[19]:


from lightgbm import LGBMClassifier
LightgbmFirst = LGBMClassifier(learning_rate = 0.1,
                               lambda_l1=0.1,
                               lambda_l2=0.2,
                               max_depth=4,
                               objective='multiclass',
                               num_class=3,
                               random_state=2022)
LightgbmFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
LightgbmFirst_score = LightgbmFirst.score(XdataOneFirst_test, ydataOneFirst_test)
LightgbmFirst_score


# #### 逻辑回归

# In[20]:


from sklearn.linear_model import LogisticRegression
LogisticRegressionFirst = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)
LogisticRegressionFirst = LogisticRegressionFirst.fit(XdataOneFirst_train, ydataOneFirst_train)
LogisticRegressionFirst_score = LogisticRegressionFirst.score(XdataOneFirst_test, ydataOneFirst_test)
LogisticRegressionFirst_score


# In[21]:


print(f'模型一中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中RF均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中KNN均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中SVM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LR均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')


# #### 集成学习

# In[22]:


from mlxtend.classifier import StackingCVClassifier
FirstModel = StackingCVClassifier(classifiers=[LogisticRegressionFirst,XGBFirst,KNNFirst_new,SVMFirst,LightgbmFirst], meta_classifier=RandomForestClassifier(random_state=2022), random_state=2022, cv=5)
FirstModel.fit(XdataOneFirst_train, ydataOneFirst_train)
FirstModel_score = FirstModel.score(XdataOneFirst_test, ydataOneFirst_test)
FirstModel_score


# In[23]:


print(f'模型一平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')


# ### "网络覆盖与信号强度"学习

# In[24]:


XdataOneSecond=dataOneNewStandard.loc[:,~dataOneNewStandard.columns.isin(['语音通话整体满意度','网络覆盖与信号强度',
                                                                          '语音通话清晰度','语音通话稳定性'])]
ydataOneSecond=dataOneNewStandard['网络覆盖与信号强度']
XdataOneSecond_train, XdataOneSecond_test, ydataOneSecond_train, ydataOneSecond_test = train_test_split(XdataOneSecond, ydataOneSecond, test_size=0.2, random_state=2022)


# #### 决策树、随机森林

# In[25]:


DecisionTreeSecond = DecisionTreeClassifier(random_state=2022)
RandomForestSecond = RandomForestClassifier(random_state=2022)
DecisionTreeSecond = DecisionTreeSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond = RandomForestSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond_score = RandomForestSecond.score(XdataOneSecond_test, ydataOneSecond_test)
RandomForestSecond_score


# In[26]:


RandomForestSecond = RandomForestClassifier(n_estimators=164, random_state=2022, min_samples_leaf=8, max_depth=19)
RandomForestSecond = RandomForestSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
RandomForestSecond_score = RandomForestSecond.score(XdataOneSecond_test, ydataOneSecond_test)
RandomForestSecond_score


# #### XGBoost

# In[27]:


from xgboost import XGBClassifier

XGBSecond = XGBClassifier(learning_rate=0.02,
                          n_estimators=13,
                          max_depth=8,
                          min_child_weight=1,
                          gamma=0.05,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=2022,
                          slient=0)
XGBSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
XGBSecond_score = XGBSecond.score(XdataOneSecond_test, ydataOneSecond_test)
XGBSecond_score


# #### KNN

# In[28]:


from sklearn.neighbors import KNeighborsClassifier

KNNSecond = KNeighborsClassifier()
KNNSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
KNNSecond_score = KNNSecond.score(XdataOneSecond_test, ydataOneSecond_test)
KNNSecond_score


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights':['uniform'],
                          'n_neighbors':[k for k in range(40,50)]},
                         {'weights':['distance'],
                          'n_neighbors':[k for k in range(40,50)],
                          'p':[p for p in range(1,5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid = KNN_turing_param_grid,
                                      n_jobs = -1,
                                      verbose = 2)
KNN_turing_grid_search.fit(XdataOneSecond_train, ydataOneSecond_train)


# In[30]:


KNN_turing_grid_search.best_score_


# In[31]:


KNN_turing_grid_search.best_params_


# In[32]:


KNNSecond_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                     metric='minkowski',
                                     n_jobs=-1,
                                     n_neighbors=43, p=1,
                                     weights='uniform')
KNNSecond_new.fit(XdataOneSecond_train, ydataOneSecond_train)
KNNSecond_new_score = KNNSecond_new.score(XdataOneSecond_test, ydataOneSecond_test)
KNNSecond_new_score


# #### 支持向量机

# In[33]:


from sklearn.svm import SVC

SVMSecond = SVC(random_state=2022)
SVMSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
SVMSecond_score = SVMSecond.score(XdataOneSecond_test, ydataOneSecond_test)
SVMSecond_score


# #### lightgbm

# In[34]:


from lightgbm import LGBMClassifier
LightgbmSecond = LGBMClassifier(learning_rate = 0.1,
                                lambda_l1=0.1,
                                lambda_l2=0.2,
                                max_depth=3,
                                objective='multiclass',
                                num_class=3,
                                random_state=2022)
LightgbmSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
LightgbmSecond_score = LightgbmSecond.score(XdataOneSecond_test, ydataOneSecond_test)
LightgbmSecond_score


# #### 逻辑回归

# In[35]:


from sklearn.linear_model import LogisticRegression
LogisticRegressionSecond = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionSecond = LogisticRegressionSecond.fit(XdataOneSecond_train, ydataOneSecond_train)
LogisticRegressionSecond_score = LogisticRegressionSecond.score(XdataOneSecond_test, ydataOneSecond_test)
LogisticRegressionSecond_score


# In[36]:


print(f'模型二中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中RF均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中KNN均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中SVM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LR均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')


# #### 集成学习

# In[37]:


from mlxtend.classifier import StackingCVClassifier
SecondModel = StackingCVClassifier(classifiers=[RandomForestSecond,XGBSecond,KNNSecond_new,SVMSecond,LogisticRegressionSecond], meta_classifier=LGBMClassifier(random_state=2022), random_state=2022, cv=5)
SecondModel.fit(XdataOneSecond_train, ydataOneSecond_train)
SecondModel_score = SecondModel.score(XdataOneSecond_test, ydataOneSecond_test)
SecondModel_score


# In[38]:


print(f'模型二平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')


# ### "语音通话清晰度"学习

# In[39]:


XdataOneThird=dataOneNewStandard.loc[:,~dataOneNewStandard.columns.isin(['语音通话整体满意度','网络覆盖与信号强度',
                                                                         '语音通话清晰度','语音通话稳定性'])]
ydataOneThird=dataOneNewStandard['语音通话清晰度']
XdataOneThird_train, XdataOneThird_test, ydataOneThird_train, ydataOneThird_test = train_test_split(XdataOneThird, ydataOneThird, test_size=0.2, random_state=2022)


# #### 决策树、随机森林

# In[40]:


DecisionTreeThird = DecisionTreeClassifier(random_state=2022)
RandomForestThird = RandomForestClassifier(random_state=2022)
DecisionTreeThird = DecisionTreeThird.fit(XdataOneThird_train, ydataOneThird_train)
RandomForestThird = RandomForestThird.fit(XdataOneThird_train, ydataOneThird_train)
RandomForestThird_score = RandomForestThird.score(XdataOneThird_test, ydataOneThird_test)
RandomForestThird_score


# #### XGBoost

# In[41]:


from xgboost import XGBClassifier

XGBThird = XGBClassifier(learning_rate=0.02,
                         n_estimators=14,
                         max_depth=8,
                         min_child_weight=1,
                         gamma=0.05,
                         subsample=1,
                         colsample_btree=1,
                         scale_pos_weight=1,
                         random_state=2022,
                         slient=0)
XGBThird.fit(XdataOneThird_train, ydataOneThird_train)
XGBThird_score = XGBThird.score(XdataOneThird_test, ydataOneThird_test)
XGBThird_score


# #### KNN

# In[42]:


from sklearn.neighbors import KNeighborsClassifier

KNNThird = KNeighborsClassifier()
KNNThird.fit(XdataOneThird_train, ydataOneThird_train)
KNNThird_score = KNNThird.score(XdataOneThird_test, ydataOneThird_test)
KNNThird_score


# In[43]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights':['uniform'],
                          'n_neighbors':[k for k in range(30,40)]},
                         {'weights':['distance'],
                          'n_neighbors':[k for k in range(30,40)],
                          'p':[p for p in range(1,5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid = KNN_turing_param_grid,
                                      n_jobs = -1,
                                      verbose = 2)
KNN_turing_grid_search.fit(XdataOneThird_train, ydataOneThird_train)


# In[44]:


KNN_turing_grid_search.best_score_


# In[45]:


KNN_turing_grid_search.best_params_


# In[46]:


KNNThird_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                    metric='minkowski',
                                    n_jobs=-1,
                                    n_neighbors=39, p=2,
                                    weights='uniform')
KNNThird_new.fit(XdataOneThird_train, ydataOneThird_train)
KNNThird_new_score = KNNThird_new.score(XdataOneThird_test, ydataOneThird_test)
KNNThird_new_score


# #### 支持向量机

# In[47]:


from sklearn.svm import SVC

SVMThird = SVC(random_state=2022)
SVMThird.fit(XdataOneThird_train, ydataOneThird_train)
SVMThird_score = SVMThird.score(XdataOneThird_test, ydataOneThird_test)
SVMThird_score


# #### lightgbm

# In[48]:


from lightgbm import LGBMClassifier
LightgbmThird = LGBMClassifier(learning_rate = 0.1,
                                lambda_l1=0.1,
                                lambda_l2=0.2,
                                max_depth=9,
                                objective='multiclass',
                                num_class=4,
                                random_state=2022)
LightgbmThird.fit(XdataOneThird_train, ydataOneThird_train)
LightgbmThird_score = LightgbmThird.score(XdataOneThird_test, ydataOneThird_test)
LightgbmThird_score


# #### 逻辑回归

# In[49]:


from sklearn.linear_model import LogisticRegression
LogisticRegressionThird = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionThird = LogisticRegressionThird.fit(XdataOneThird_train, ydataOneThird_train)
LogisticRegressionThird_score = LogisticRegressionThird.score(XdataOneThird_test, ydataOneThird_test)
LogisticRegressionThird_score


# In[50]:


print(f'模型三中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中RF均方误差：'
      f'{mean_squared_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中KNN均方误差：'
      f'{mean_squared_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中SVM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LR均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')


# #### 集成学习

# In[51]:


from mlxtend.classifier import StackingCVClassifier
ThirdModel = StackingCVClassifier(classifiers=[XGBThird,LightgbmThird,KNNThird_new,SVMThird,LogisticRegressionThird], meta_classifier=RandomForestClassifier(random_state=2022), random_state=2022, cv=5)
ThirdModel.fit(XdataOneThird_train, ydataOneThird_train)
ThirdModel_score = ThirdModel.score(XdataOneThird_test, ydataOneThird_test)
ThirdModel_score


# In[52]:


print(f'模型三平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三均方误差：'
      f'{mean_squared_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')


# ### "语音通话稳定性"学习

# In[53]:


XdataOneFourth=dataOneNewStandard.loc[:,~dataOneNewStandard.columns.isin(['语音通话整体满意度','网络覆盖与信号强度',
                                                                          '语音通话清晰度','语音通话稳定性'])]
ydataOneFourth=dataOneNewStandard['语音通话稳定性']
XdataOneFourth_train, XdataOneFourth_test, ydataOneFourth_train, ydataOneFourth_test = train_test_split(XdataOneFourth, ydataOneFourth, test_size=0.2, random_state=2022)


# #### 决策树、随机森林

# In[54]:


DecisionTreeFourth = DecisionTreeClassifier(random_state=2022)
RandomForestFourth = RandomForestClassifier(random_state=2022)
DecisionTreeFourth = DecisionTreeFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
RandomForestFourth = RandomForestFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
RandomForestFourth_score = RandomForestFourth.score(XdataOneFourth_test, ydataOneFourth_test)
RandomForestFourth_score


# #### XGBoost

# In[55]:


from xgboost import XGBClassifier

XGBFourth = XGBClassifier(learning_rate=0.02,
                          n_estimators=14,
                          max_depth=6,
                          min_child_weight=1,
                          gamma=0.05,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=2022,
                          slient=0)
XGBFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
XGBFourth_score = XGBFourth.score(XdataOneFourth_test, ydataOneFourth_test)
XGBFourth_score


# #### KNN

# In[56]:


from sklearn.neighbors import KNeighborsClassifier

KNNFourth = KNeighborsClassifier()
KNNFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
KNNFourth_score = KNNFourth.score(XdataOneFourth_test, ydataOneFourth_test)
KNNFourth_score


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN_turing_param_grid = [{'weights':['uniform'],
                          'n_neighbors':[k for k in range(35,45)]},
                         {'weights':['distance'],
                          'n_neighbors':[k for k in range(35,45)],
                          'p':[p for p in range(1,5)]}]
KNN_turing = KNeighborsClassifier()
KNN_turing_grid_search = GridSearchCV(KNN_turing,
                                      param_grid = KNN_turing_param_grid,
                                      n_jobs = -1,
                                      verbose = 2)
KNN_turing_grid_search.fit(XdataOneFourth_train, ydataOneFourth_train)


# In[58]:


KNN_turing_grid_search.best_score_


# In[59]:


KNN_turing_grid_search.best_params_


# In[60]:


KNNFourth_new = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                     metric='minkowski',
                                     n_jobs=-1,
                                     n_neighbors=41, p=1,
                                     weights='distance')
KNNFourth_new.fit(XdataOneFourth_train, ydataOneFourth_train)
KNNFourth_new_score = KNNFourth_new.score(XdataOneFourth_test, ydataOneFourth_test)
KNNFourth_new_score


# #### 支持向量机

# In[61]:


from sklearn.svm import SVC

SVMFourth = SVC(random_state=2022)
SVMFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
SVMFourth_score = SVMFourth.score(XdataOneFourth_test, ydataOneFourth_test)
SVMFourth_score


# #### lightgbm

# In[62]:


from lightgbm import LGBMClassifier
LightgbmFourth = LGBMClassifier(learning_rate = 0.1,
                                lambda_l1=0.1,
                                lambda_l2=0.2,
                                max_depth=10,
                                objective='multiclass',
                                num_class=4,
                                random_state=2022)
LightgbmFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
LightgbmFourth_score = LightgbmFourth.score(XdataOneFourth_test, ydataOneFourth_test)
LightgbmFourth_score


# #### 逻辑回归

# In[63]:


from sklearn.linear_model import LogisticRegression
LogisticRegressionFourth = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=2000)
LogisticRegressionFourth = LogisticRegressionFourth.fit(XdataOneFourth_train, ydataOneFourth_train)
LogisticRegressionFourth_score = LogisticRegressionFourth.score(XdataOneFourth_test, ydataOneFourth_test)
LogisticRegressionFourth_score


# In[64]:


print(f'模型四中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中RF均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中KNN均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中SVM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LR均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')


# #### 集成学习

# In[65]:


from mlxtend.classifier import StackingCVClassifier
FourthModel = StackingCVClassifier(classifiers=[RandomForestFourth,LightgbmFourth,KNNFourth_new,LogisticRegressionFourth,SVMFourth], meta_classifier=XGBClassifier(random_state=2022), random_state=2022, cv=5)
FourthModel.fit(XdataOneFourth_train, ydataOneFourth_train)
FourthModel_score = FourthModel.score(XdataOneFourth_test, ydataOneFourth_test)
FourthModel_score


# In[66]:


print(f'模型四平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')


# ## 预测附件3四项评分

# In[67]:


dataThree=pd.read_excel("附件3语音业务用户满意度预测数据.xlsx",sheet_name='语音')
dataThree


# ### 附件格式统一

# In[68]:


dataThree.drop(['用户id',
                '用户描述',
                '用户描述.1',
                '性别',
                '终端品牌类型',
                '是否不限量套餐到达用户'], axis=1, inplace=True)


# In[69]:


dataThree


# In[70]:


dataThree.isnull().sum()


# In[71]:


dataThree["外省流量占比"] = dataThree["外省流量占比"].astype(str).replace('%','')
dataThree["外省语音占比"] = dataThree["外省语音占比"].astype(str).replace('%','')
dataThree


# In[72]:


dataThree.replace({"是否遇到过网络问题":{2:0},
                   "居民小区":{-1:0},
                   "办公室":{-1:0,2:1},
                   "高校":{-1:0,3:1},
                   "商业街":{-1:0,4:1},
                   "地铁":{-1:0,5:1},
                   "农村":{-1:0,6:1},
                   "高铁":{-1:0,7:1},
                   "其他，请注明":{-1:0,98:1},
                   "手机没有信号":{-1:0},
                   "有信号无法拨通":{-1:0,2:1},
                   "通话过程中突然中断":{-1:0,3:1},
                   "通话中有杂音、听不清、断断续续":{-1:0,4:1},
                   "串线":{-1:0,5:1},
                   "通话过程中一方听不见":{-1:0,6:1},
                   "其他，请注明.1":{-1:0,98:1},
                   "是否投诉":{'是':1,'否':0},
                   "是否关怀用户":{'是':1,'否':0},
                   "是否4G网络客户（本地剔除物联网）":{'是':1,"否":0},
                   "是否5G网络客户":{'是':1,"否":0},
                   "客户星级标识":{'未评级':0,'准星':1,'一星':2,'二星':3,'三星':4,'银卡':5,'金卡':6,'白金卡':7,'钻石卡':8},
                   "终端品牌":{'苹果':22,'华为':11,'小米科技':14,
                            '步步高':18,'欧珀':17,'三星':4,
                            'realme':1,'0':0,'万普拉斯':3,
                            '锤子':24,'万普':8,'中邮通信':21,
                            '索尼爱立信':6,'亿城':6,'宇龙':6,
                            '中国移动':7,'中兴':10,'黑鲨':25,
                            '海信':16,'摩托罗拉':9,'诺基亚':12,
                            '奇酷':13}
                   }, inplace=True)
dataThree


# In[73]:


dataThree['外省语音占比'] = dataThree['外省语音占比'].astype('float64')
dataThree['外省流量占比'] = dataThree['外省流量占比'].astype('float64')
dataThree['是否4G网络客户（本地剔除物联网）'] = dataThree['是否4G网络客户（本地剔除物联网）'].astype('int64')
dataThree['4\\5G用户'] = dataThree['4\\5G用户'].astype(str)
dataThree


# In[74]:


le=sp.LabelEncoder()

FourFiveUser=le.fit_transform(dataThree["4\\5G用户"])
dataThree["4\\5G用户"]=pd.DataFrame(FourFiveUser)
dataThree


# In[75]:


dataThree['是否5G网络客户'] = dataThree['是否5G网络客户'].astype('int64')
dataThree['客户星级标识'] = dataThree['客户星级标识'].astype('int64')
dataThree['终端品牌'] = dataThree['终端品牌'].astype('int32')
dataThree


# In[76]:


dataThree['场所合计']=dataThree.loc[:,['居民小区','办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明']].apply(lambda x1:x1.sum(),axis=1)
dataThree['出现问题合计']=dataThree.loc[:,['手机没有信号','有信号无法拨通','通话过程中突然中断','通话中有杂音、听不清、断断续续','串线','通话过程中一方听不见','其他，请注明.1']].apply(lambda x1:x1.sum(),axis=1)
dataThree['脱网次数、mos质差次数、未接通掉话次数合计']=dataThree.loc[:,['脱网次数','mos质差次数','未接通掉话次数']].apply(lambda x1:x1.sum(),axis=1)
dataThree


# In[77]:


dataThreeStandardTransform = dataThree[['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                                        '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                                        '终端品牌','当月ARPU','当月MOU',
                                        '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                                        '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']]
dataThreeStandardTransformScaler = sp.StandardScaler()
dataThreeStandardTransformScaler = dataThreeStandardTransformScaler.fit(dataThreeStandardTransform)
dataThreeStandardTransform = dataThreeStandardTransformScaler.transform(dataThreeStandardTransform)
dataThreeStandardTransform = pd.DataFrame(dataThreeStandardTransform)
dataThreeStandardTransform.columns = ['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                                      '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                                      '终端品牌','当月ARPU','当月MOU',
                                      '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                                      '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']
dataThreeStandardTransform


# In[78]:


dataThreeLeave=dataThree.loc[:,~dataThree.columns.isin(['脱网次数','mos质差次数','未接通掉话次数','4\\5G用户',
                                                        '套外流量（MB）','套外流量费（元）','语音通话-时长（分钟）','省际漫游-时长（分钟）',
                                                        '终端品牌','当月ARPU','当月MOU',
                                                        '前3月ARPU','前3月MOU','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
                                                        '客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计'])]
dataThreeNewStandard=pd.concat([dataThreeLeave, dataThreeStandardTransform],axis=1)
dataThreeNewStandard.columns=['是否遇到过网络问题','居民小区','办公室','高校',
                              '商业街','地铁','农村','高铁',
                              '其他，请注明','手机没有信号','有信号无法拨通','通话过程中突然中断',
                              '通话中有杂音、听不清、断断续续','串线','通话过程中一方听不见','其他，请注明.1',
                              '是否投诉','是否关怀用户','是否4G网络客户（本地剔除物联网）','外省语音占比',
                              '外省流量占比','是否5G网络客户','脱网次数','mos质差次数',
                              '未接通掉话次数','4\\5G用户','套外流量（MB）','套外流量费（元）',
                              '语音通话-时长（分钟）','省际漫游-时长（分钟）','终端品牌',
                              '当月ARPU','当月MOU','前3月ARPU','前3月MOU',
                              'GPRS总流量（KB）','GPRS-国内漫游-流量（KB）','客户星级标识','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']
dataThreeNewStandard


# In[79]:


dataOneNewStandard


# ### 预测语音业务评分
# 需要注意到在所有预测结果上加上1，由于之前将评分编码为[0,9]，这里需要再映射回[1,10]

# In[80]:


Xpre=dataThreeNewStandard


# #### 语音通话整体满意度

# In[81]:


FirstPre=FirstModel.predict(Xpre)
FirstPre


# #### 网络覆盖与信号强度

# In[82]:


SecondPre=SecondModel.predict(Xpre)
SecondPre


# #### 语音通话清晰度

# In[83]:


ThirdPre=ThirdModel.predict(Xpre)
ThirdPre


# #### 语音通话稳定性

# In[84]:


FourthPre=FourthModel.predict(Xpre)
FourthPre


# ## 模型效果分析

# In[85]:


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# ### 混淆矩阵热力图

# #### 模型一

# In[86]:


from yellowbrick.classifier import ConfusionMatrix
classes=['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = ConfusionMatrix(FirstModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneFirst_train, ydataOneFirst_train)
confusion_matrix.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型一混淆矩阵热力图.pdf')


# #### 模型二

# In[87]:


from yellowbrick.classifier import ConfusionMatrix
classes=['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = ConfusionMatrix(SecondModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneSecond_train, ydataOneSecond_train)
confusion_matrix.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型二混淆矩阵热力图.pdf')


# #### 模型三

# In[88]:


from yellowbrick.classifier import ConfusionMatrix
classes=['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = ConfusionMatrix(ThirdModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneThird_train, ydataOneThird_train)
confusion_matrix.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型三混淆矩阵热力图.pdf')


# #### 模型四

# In[89]:


from yellowbrick.classifier import ConfusionMatrix
classes=['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = ConfusionMatrix(FourthModel, classes=classes, cmap='BuGn')
confusion_matrix.fit(XdataOneFourth_train, ydataOneFourth_train)
confusion_matrix.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figuresOne\\[附件1]模型四混淆矩阵热力图.pdf')


# ### 分类报告

# #### 模型一

# In[90]:


from yellowbrick.classifier import ClassificationReport
classes=['1','2','3','4','5','6','7','8','9','10']
visualizer = ClassificationReport(FirstModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneFirst_train, ydataOneFirst_train)
visualizer.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型一分类报告.pdf')


# #### 模型二

# In[91]:


from yellowbrick.classifier import ClassificationReport
classes=['1','2','3','4','5','6','7','8','9','10']
visualizer = ClassificationReport(SecondModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneSecond_train, ydataOneSecond_train)
visualizer.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型二分类报告.pdf')


# #### 模型三

# In[92]:


from yellowbrick.classifier import ClassificationReport
classes=['1','2','3','4','5','6','7','8','9','10']
visualizer = ClassificationReport(ThirdModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneThird_train, ydataOneThird_train)
visualizer.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型三分类报告.pdf')


# #### 模型四

# In[93]:


from yellowbrick.classifier import ClassificationReport
classes=['1','2','3','4','5','6','7','8','9','10']
visualizer = ClassificationReport(FourthModel, classes=classes, support=True, cmap='Blues')
visualizer.fit(XdataOneFourth_train, ydataOneFourth_train)
visualizer.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型四分类报告.pdf')


# ### ROC AUC曲线

# #### 模型一

# In[94]:


from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(FirstModel)
visualizer.fit(XdataOneFirst_train, ydataOneFirst_train)
visualizer.score(XdataOneFirst_test, ydataOneFirst_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型一ROCAUC.pdf')


# #### 模型二

# In[95]:


from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(SecondModel)
visualizer.fit(XdataOneSecond_train, ydataOneSecond_train)
visualizer.score(XdataOneSecond_test, ydataOneSecond_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型二ROCAUC.pdf')


# #### 模型三

# In[96]:


from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(ThirdModel)
visualizer.fit(XdataOneThird_train, ydataOneThird_train)
visualizer.score(XdataOneThird_test, ydataOneThird_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型三ROCAUC.pdf')


# #### 模型四

# In[97]:


from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(FourthModel)
visualizer.fit(XdataOneFourth_train, ydataOneFourth_train)
visualizer.score(XdataOneFourth_test, ydataOneFourth_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figuresOne\\[附件1]模型四ROCAUC.pdf')


# ### 平均绝对误差，均方误差

# #### 模型一

# In[98]:


print(f'模型一平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, FirstModel.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中RF均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, RandomForestFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, XGBFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中KNN均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, KNNFirst_new.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中SVM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, SVMFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LightgbmFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型一中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型一中LR均方误差：'
      f'{mean_squared_error(ydataOneFirst_test, LogisticRegressionFirst.predict(XdataOneFirst_test), sample_weight=None, multioutput="uniform_average")}')


# #### 模型二

# In[99]:


print(f'模型二平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SecondModel.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中RF均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, RandomForestSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, XGBSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中KNN均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, KNNSecond_new.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中SVM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, SVMSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LightgbmSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型二中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型二中LR均方误差：'
      f'{mean_squared_error(ydataOneSecond_test, LogisticRegressionSecond.predict(XdataOneSecond_test), sample_weight=None, multioutput="uniform_average")}')


# #### 模型三

# In[100]:


print(f'模型三平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三均方误差：'
      f'{mean_squared_error(ydataOneThird_test, ThirdModel.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中RF均方误差：'
      f'{mean_squared_error(ydataOneThird_test, RandomForestThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneThird_test, XGBThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中KNN均方误差：'
      f'{mean_squared_error(ydataOneThird_test, KNNThird_new.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中SVM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, SVMThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LightgbmThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型三中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型三中LR均方误差：'
      f'{mean_squared_error(ydataOneThird_test, LogisticRegressionThird.predict(XdataOneThird_test), sample_weight=None, multioutput="uniform_average")}')


# #### 模型四

# In[101]:


print(f'模型四平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, FourthModel.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中RF平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中RF均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, RandomForestFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中XGBoost平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中XGBoost均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, XGBFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中KNN平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中KNN均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, KNNFourth_new.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中SVM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中SVM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, SVMFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LightGBM平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LightGBM均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LightgbmFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')
print(f'模型四中LR平均绝对误差：'
      f'{mean_absolute_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'模型四中LR均方误差：'
      f'{mean_squared_error(ydataOneFourth_test, LogisticRegressionFourth.predict(XdataOneFourth_test), sample_weight=None, multioutput="uniform_average")}')


# ## 高频词汇云图

# In[102]:


import jieba
import wordcloud
from matplotlib.image import imread

jieba.setLogLevel(jieba.logging.INFO)
report = open('语音业务词云.txt', 'r', encoding='utf-8').read()
words = jieba.lcut(report)
txt = []
for word in words:
    if len(word) == 1:
        continue
    else:
        txt.append(word)
a = ' '.join(txt)
bg = imread("bg.jpg")
w = wordcloud.WordCloud(background_color="white", font_path="msyh.ttc", mask=bg)
w.generate(a)
w.to_file("figuresOne\\wordcloudF.png")

