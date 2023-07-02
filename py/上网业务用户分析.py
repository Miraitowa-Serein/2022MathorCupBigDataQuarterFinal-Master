#!/usr/bin/env python
# coding: utf-8

# # 上网业务 用户行为分析

# ## 导入库

# In[1]:


import pandas as pd
import seaborn as sns


# ## 数据预处理

# In[2]:


dataTwo=pd.read_excel("附件2上网业务用户满意度数据.xlsx",sheet_name='用后即评满意度分析0620(Q1655704201796)_P')
dataFour=pd.read_excel("附件4上网业务用户满意度预测数据.xlsx",sheet_name='上网')


# In[3]:


dataTwo


# In[4]:


dataFour


# In[5]:


list(set(list(dataTwo.columns))&set(list(dataFour.columns)))


# In[6]:


dataTwo=dataTwo[['手机上网整体满意度','网络覆盖与信号强度','手机上网速度','手机上网稳定性',
                 '居民小区','是否5G网络客户','高校',
                 '是否不限量套餐到达用户','其他，请注明.5','咪咕视频','阴阳师',
                 '手机QQ','手机上网速度慢','炉石传说','打游戏延时大',
                 '火山','显示有信号上不了网','今日头条','办公室',
                 '上网质差次数','梦幻西游','当月MOU','其他，请注明.2',
                 '客户星级标识','穿越火线','全部都卡顿','微信',
                 '全部游戏都卡顿','脱网次数','性别','套外流量费（元）',
                 '农村','搜狐视频','京东','微信质差次数',
                 '百度','套外流量（MB）','其他，请注明.1','抖音',
                 '商业街','拼多多','新浪微博','其他，请注明',
                 '和平精英','手机支付较慢','看视频卡顿','终端品牌',
                 '梦幻诛仙','部落冲突','腾讯视频','上网过程中网络时断时续或时快时慢',
                 '其他，请注明.3','地铁','打开网页或APP图片慢','快手',
                 '芒果TV','爱奇艺','龙之谷','高铁',
                 '全部网页或APP都慢','王者荣耀','淘宝','其他，请注明.4',
                 '下载速度慢','优酷','欢乐斗地主','网络信号差/没有信号']]
dataTwo


# In[7]:


dataFour=dataFour[['居民小区','是否5G网络客户','高校',
                   '是否不限量套餐到达用户','其他，请注明.5','咪咕视频','阴阳师',
                   '手机QQ','手机上网速度慢','炉石传说','打游戏延时大',
                   '火山','显示有信号上不了网','今日头条','办公室',
                   '上网质差次数','梦幻西游','当月MOU','其他，请注明.2',
                   '客户星级标识','穿越火线','全部都卡顿','微信',
                   '全部游戏都卡顿','脱网次数','性别','套外流量费（元）',
                   '农村','搜狐视频','京东','微信质差次数',
                   '百度','套外流量（MB）','其他，请注明.1','抖音',
                   '商业街','拼多多','新浪微博','其他，请注明',
                   '和平精英','手机支付较慢','看视频卡顿','终端品牌',
                   '梦幻诛仙','部落冲突','腾讯视频','上网过程中网络时断时续或时快时慢',
                   '其他，请注明.3','地铁','打开网页或APP图片慢','快手',
                   '芒果TV','爱奇艺','龙之谷','高铁',
                   '全部网页或APP都慢','王者荣耀','淘宝','其他，请注明.4',
                   '下载速度慢','优酷','欢乐斗地主','网络信号差/没有信号']]
dataFour


# In[8]:


dataTwo=dataTwo.fillna(0)
dataTwo


# In[9]:


dataTwo.replace({'居民小区':{-1:0},
                 '是否5G网络客户':{'否':0,'是':1},
                 '高校':{-1:0,3:1},
                 '是否不限量套餐到达用户':{'否':0,'是':1},
                 '其他，请注明.5':{-1:0,98:1},
                 '咪咕视频':{-1:0,9:1},
                 '阴阳师':{-1:0,10:1},
                 '手机QQ':{-1:0,2:1},
                 '手机上网速度慢':{-1:0,4:1},
                 '炉石传说':{-1:0,9:1},
                 '打游戏延时大':{-1:0,2:1},
                 '火山':{-1:0,8:1},
                 '显示有信号上不了网':{-1:0,2:1},
                 '今日头条':{-1:0,6:1},
                 '办公室':{-1:0,2:1},
                 '梦幻西游':{-1:0,4:1},
                 '其他，请注明.2':{-1:0,98:1},
                 '客户星级标识':{'未评级':0,'准星':1,'一星':2,'二星':3,'三星':4,'银卡':5,'金卡':6,'白金卡':7,'钻石卡':8},
                 '穿越火线':{-1:0,3:1},
                 '全部都卡顿':{-1:0,99:1},
                 '微信':{-1:0},
                 '全部游戏都卡顿':{-1:0,99:1},
                 '性别':{'男':1,'女':-1,'性别不详':0},
                 '农村':{-1:0,6:1},
                 '搜狐视频':{-1:0,5:1},
                 '京东':{-1:0,4:1},
                 '百度':{-1:0,5:1},
                 '其他，请注明.1':{-1:0,98:1},
                 '抖音':{-1:0,6:1},
                 '商业街':{-1:0,4:1},
                 '拼多多':{-1:0,8:1},
                 '新浪微博':{-1:0,7:1},
                 '其他，请注明':{-1:0,98:1},
                 '和平精英':{-1:0},
                 '手机支付较慢':{-1:0,5:1},
                 '看视频卡顿':{-1:0},
                 '梦幻诛仙':{-1:0,6:1},
                 '部落冲突':{-1:0,8:1},
                 '腾讯视频':{-1:0,3:1},
                 '上网过程中网络时断时续或时快时慢':{-1:0,3:1},
                 '其他，请注明.3':{-1:0,98:1},
                 '地铁':{-1:0,5:1},
                 '打开网页或APP图片慢':{-1:0,3:1},
                 '快手':{-1:0,7:1},
                 '芒果TV':{-1:0,4:1},
                 '爱奇艺':{-1:0},
                 '龙之谷':{-1:0,5:1},
                 '高铁':{-1:0,7:1},
                 '全部网页或APP都慢':{-1:0,99:1},
                 '王者荣耀':{-1:0,2:1},
                 '淘宝':{-1:0,3:1},
                 '其他，请注明.4':{-1:0,98:1},
                 '下载速度慢':{-1:0,4:1},
                 '优酷':{-1:0,2:1},
                 '欢乐斗地主':{-1:0,7:1},
                 '网络信号差/没有信号':{-1:0},
                 '终端品牌':{'0':0,'苹果':1,'华为':2,'小米科技':3,
                            '步步高':4,'欧珀':5,'realme':6,'三星':7,
                            '万普拉斯':8,'黑鲨':9,'锤子':10,'摩托罗拉':11,
                            '中邮通信':12,'万普':13,'诺基亚':14,'联通':15,
                            '中国移动':16,'中兴':17,'华硕':18,'联想':19,
                            '魅族':20,'奇酷':21,'TD':22,'北京珠穆朗玛移动通信有限公司':23,
                            '飞利浦':24,'捷开通讯科技':25,'金立':26,'酷比':27,
                            '欧博信':28,'索尼爱立信':29,'维图':30,'甄十信息科技（上海）有限公司':31,
                            '中国电信':32}
                 }, inplace=True)
dataTwo


# In[10]:


import sklearn.preprocessing as sp
le=sp.LabelEncoder()

OverallSatisfactionMobileInternetAccess=le.fit_transform(dataTwo['手机上网整体满意度'])
NetworkCoverageSignalStrength=le.fit_transform(dataTwo['网络覆盖与信号强度'])
MobileInternetAccessSpeed=le.fit_transform(dataTwo['手机上网速度'])
MobileInternetAccessStability=le.fit_transform(dataTwo['手机上网稳定性'])

dataTwo["手机上网整体满意度"]=pd.DataFrame(OverallSatisfactionMobileInternetAccess)
dataTwo["网络覆盖与信号强度"]=pd.DataFrame(NetworkCoverageSignalStrength)
dataTwo["手机上网速度"]=pd.DataFrame(MobileInternetAccessSpeed)
dataTwo["手机上网稳定性"]=pd.DataFrame(MobileInternetAccessStability)

dataTwo


# In[11]:


dataTwo['出现问题场所或应用总']=dataTwo.loc[:,~dataTwo.columns.isin(['手机上网整体满意度','网络覆盖与信号强度','手机上网速度','手机上网稳定性',
                            '是否5G网络客户','是否不限量套餐到达用户','手机上网速度慢','打游戏延时大',
                            '显示有信号上不了网','上网质差次数','当月MOU','客户星级标识',
                            '全部都卡顿','全部游戏都卡顿','脱网次数','性别',
                            '套外流量费（元）','微信质差次数','百度','套外流量（MB）',
                            '手机支付较慢','看视频卡顿','终端品牌','上网过程中网络时断时续或时快时慢',
                            '打开网页或APP图片慢','全部网页或APP都慢','下载速度慢','网络信号差/没有信号'])].apply(lambda x1:x1.sum(), axis=1)
dataTwo['网络卡速度慢延时大上不了网总']=dataTwo.loc[:,['手机上网速度慢','打游戏延时大','显示有信号上不了网','全部都卡顿',
                                                '全部游戏都卡顿','手机支付较慢','看视频卡顿','上网过程中网络时断时续或时快时慢',
                                                '打开网页或APP图片慢','全部网页或APP都慢',
                                                '下载速度慢','网络信号差/没有信号']].apply(lambda x1:x1.sum(), axis=1)
dataTwo['质差总']=dataTwo.loc[:,['微信质差次数','上网质差次数']].apply(lambda x1:x1.sum(), axis=1)
dataTwo['地点总']=dataTwo.loc[:,['居民小区','高校','办公室','农村','商业街','地铁','高铁']].apply(lambda x1:x1.sum(),axis=1)
dataTwo['整体评分']=dataTwo.loc[:,['手机上网整体满意度','网络覆盖与信号强度','手机上网速度','手机上网稳定性']].apply(lambda x1:round(x1.mean()),axis=1)
dataTwo


# ## 用户行为分析

# In[12]:


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

box_data = dataTwo[['手机上网整体满意度',
                    '网络覆盖与信号强度',
                    '手机上网速度',
                    '手机上网稳定性',]]
plt.grid(True)
plt.boxplot(box_data,
            notch = True,
            sym = "b+",
            vert = False,
            showmeans = True,
            labels = ['手机上网整体满意度',
                      '网络覆盖与信号强度',
                      '手机上网速度',
                      '手机上网稳定性',])
plt.yticks(size=14)
plt.xticks(size=14, font='Times New Roman')
plt.tight_layout()
plt.savefig('figuresTwo\\[附件2][手机上网整体满意度、网络覆盖与信号强度、手机上网速度、手机上网稳定性]评分箱线图.pdf')


# In[13]:


sns.pairplot(dataTwo[['手机上网整体满意度','网络覆盖与信号强度','手机上网速度','手机上网稳定性']],kind='scatter',diag_kind='kde')
plt.savefig('figuresTwo\\[附件2][手机上网整体满意度、网络覆盖与信号强度、手机上网速度、手机上网稳定性]评分联合分布图.pdf',bbox_inches='tight')


# ## 划分高分组和低分组

# In[14]:


dataTwoHigh = dataTwo[(dataTwo['手机上网整体满意度']>=7)&(dataTwo['网络覆盖与信号强度']>=7)&(dataTwo['手机上网速度']>=7)&(dataTwo['手机上网稳定性']>=7)]
dataTwoLow = dataTwo[(dataTwo['手机上网整体满意度']<=4)&(dataTwo['网络覆盖与信号强度']<=4)&(dataTwo['手机上网速度']<=4)&(dataTwo['手机上网稳定性']<=4)]


# In[15]:


dataTwoHigh.describe()


# In[16]:


dataTwoLow.describe()


# ## 特征分析

# In[17]:


sns.pairplot(dataTwoHigh[['整体评分','出现问题场所或应用总','网络卡速度慢延时大上不了网总','质差总','地点总']],kind='scatter',diag_kind='kde')
plt.savefig('figuresTwo\\[附件2]高分组[出现问题场所或应用总、网络卡速度慢延时大上不了网总、质差总、地点总]评分多变量联合分布图.pdf',bbox_inches='tight')


# In[18]:


sns.pairplot(dataTwoLow[['整体评分','出现问题场所或应用总','网络卡速度慢延时大上不了网总','质差总','地点总']],kind='scatter',diag_kind='kde')
plt.savefig('figuresTwo\\[附件2]低分组[出现问题场所或应用总、网络卡速度慢延时大上不了网总、质差总、地点总]评分多变量联合分布图.pdf',bbox_inches='tight')


# In[19]:


sns.jointplot(x='出现问题场所或应用总', y='整体评分', data=dataTwoHigh, kind='hex')
plt.savefig('figuresTwo\\[附件2]高分组出现问题场所或应用总分布.pdf',bbox_inches='tight')


# In[20]:


sns.jointplot(x='出现问题场所或应用总', y='整体评分', data=dataTwoLow, kind='hex',color='r')
plt.savefig('figuresTwo\\[附件2]低分组出现问题场所或应用总分布.pdf',bbox_inches='tight')


# In[21]:


sns.jointplot(x='网络卡速度慢延时大上不了网总', y='整体评分', data=dataTwoHigh, kind='hex')
plt.savefig('figuresTwo\\[附件2]高分组网络卡速度慢延时大上不了网总分布.pdf',bbox_inches='tight')


# In[22]:


sns.jointplot(x='网络卡速度慢延时大上不了网总', y='整体评分', data=dataTwoLow, kind='hex',color='r')
plt.savefig('figuresTwo\\[附件2]低分组网络卡速度慢延时大上不了网总分布.pdf',bbox_inches='tight')


# In[23]:


sns.jointplot(x='质差总', y='整体评分', data=dataTwoHigh, kind='hex')
plt.savefig('figuresTwo\\[附件2]高分组质差总分布.pdf',bbox_inches='tight')


# In[24]:


sns.jointplot(x='质差总', y='整体评分', data=dataTwoLow, kind='hex',color='r')
plt.savefig('figuresTwo\\[附件2]低分组质差总分布.pdf',bbox_inches='tight')


# In[25]:


sns.jointplot(x='地点总', y='整体评分', data=dataTwoHigh, kind='hex')
plt.savefig('figuresTwo\\[附件2]高分组地点总分布.pdf',bbox_inches='tight')


# In[26]:


sns.jointplot(x='地点总', y='整体评分', data=dataTwoLow, kind='hex',color='r')
plt.savefig('figuresTwo\\[附件2]低分组地点总分布.pdf',bbox_inches='tight')


# In[27]:


dataTwoHigh['终端品牌'].mode()


# In[28]:


dataTwoLow['终端品牌'].mode()


# ## 异常用户评分数据剔除

# In[29]:


dataTwoSample=dataTwo[((dataTwo['其他，请注明']==1)|(dataTwo['其他，请注明.1']==1)|(dataTwo['其他，请注明.2']==1)|(dataTwo['其他，请注明.3']==1)|(dataTwo['其他，请注明.4']==1)|(dataTwo['其他，请注明.5']==1))|((abs(dataTwo['手机上网整体满意度']-dataTwo['网络覆盖与信号强度'])<=5)&(abs(dataTwo['手机上网整体满意度']-dataTwo['手机上网速度'])<=4)&(abs(dataTwo['手机上网整体满意度']-dataTwo['手机上网稳定性'])<=4)&(dataTwo['网络覆盖与信号强度']-dataTwo['手机上网速度']<=4)&(dataTwo['网络覆盖与信号强度']-dataTwo['手机上网稳定性']<=4)&(dataTwo['手机上网速度']-dataTwo['手机上网稳定性']<=3))]
dataTwoSample


# In[30]:


dataTwo


# In[31]:


sns.heatmap(dataTwo[['手机上网整体满意度','网络覆盖与信号强度','手机上网速度','手机上网稳定性']].corr(method='pearson'),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
plt.title('上网业务评分皮尔逊相关系数热力图')
plt.savefig('figuresTwo\\[附件2]上网业务评分皮尔逊相关系数热力图.pdf',bbox_inches='tight')

