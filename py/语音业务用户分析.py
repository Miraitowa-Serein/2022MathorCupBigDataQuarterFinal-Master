#!/usr/bin/env python
# coding: utf-8

# # 语音业务 用户行为分析

# ## 导入库

# In[1]:


import pandas as pd
import seaborn as sns


# ## 数据预处理

# In[2]:


dataOne=pd.read_excel("附件1语音业务用户满意度数据.xlsx",sheet_name='Sheet1')
dataThree=pd.read_excel("附件3语音业务用户满意度预测数据.xlsx",sheet_name='语音')


# In[3]:


dataOneColumnsList=list(dataOne.columns)
dataThreeColumnsList=list(dataThree.columns)


# In[4]:


dataOneColumnsList


# In[5]:


dataThreeColumnsList


# In[6]:


set(dataOneColumnsList)&set(dataThreeColumnsList)


# In[7]:


dataOne['资费投诉']=dataOne.loc[:, ['家宽投诉','资费投诉']].apply(lambda x1:x1.sum(), axis=1)
dataOne.drop(['家宽投诉'], axis=1, inplace=True)
dataOne.rename(columns={'资费投诉':'是否投诉'}, inplace=True)
dataOne


# In[8]:


dataOneColumnsList=list(dataOne.columns)
dataOneColumnsList


# In[9]:


dataThreeColumnsList=list(dataThree.columns)
dataThreeColumnsList


# In[10]:


set(dataOneColumnsList)-set(dataThreeColumnsList)


# In[11]:


dataOne.drop(['用户id',
              '用户描述',
              '用户描述.1',
              '重定向次数',
              '重定向驻留时长',
              '语音方式',
              '是否去过营业厅',
              'ARPU（家庭宽带）',
              '是否实名登记用户',
              '当月欠费金额',
              '前第3个月欠费金额',
              '终端品牌类型'], axis=1, inplace=True)
dataOne


# In[12]:


dataOne.info()


# In[13]:


dataOne.isnull().sum()


# In[14]:


dataOne['外省流量占比']=dataOne['外省流量占比'].fillna(0)
dataOne["是否关怀用户"]=dataOne["是否关怀用户"].fillna(0)
dataOne["外省流量占比"]=dataOne["外省流量占比"].astype(str).replace('%','')
dataOne["外省语音占比"]=dataOne["外省语音占比"].astype(str).replace('%','')
dataOne


# In[15]:


dataOne.replace({"是否遇到过网络问题":{2:0},
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
                 "是否关怀用户":{'是':1},
                 "是否4G网络客户（本地剔除物联网）":{'是':1,"否":0},
                 "是否5G网络客户":{'是':1,"否":0},
                 "客户星级标识":{'未评级':0,'准星':1,'一星':2,'二星':3,'三星':4,'银卡':5,'金卡':6,'白金卡':7,'钻石卡':8}
                 }, inplace=True)
dataOne


# In[16]:


dataOne.isnull().sum()


# In[17]:


dataOneMiss=dataOne.isnull()
dataOne[dataOneMiss.any(axis=1)==True]


# In[18]:


dataOne.dropna(inplace=True)
dataOne=dataOne.reset_index(drop=True)
dataOne


# In[19]:


dataOne.dtypes


# In[20]:


dataOne['外省语音占比'] = dataOne['外省语音占比'].astype('float64')
dataOne['外省流量占比'] = dataOne['外省流量占比'].astype('float64')
dataOne['是否4G网络客户（本地剔除物联网）'] = dataOne['是否4G网络客户（本地剔除物联网）'].astype('int64')
dataOne['4\\5G用户'] = dataOne['4\\5G用户'].astype(str)
dataOne['终端品牌'] = dataOne['终端品牌'].astype(str)
dataOne


# In[21]:


import sklearn.preprocessing as sp
le=sp.LabelEncoder()

OverallSatisfactionVoiceCalls=le.fit_transform(dataOne["语音通话整体满意度"])
NetworkCoverageSignalStrength=le.fit_transform(dataOne["网络覆盖与信号强度"])
VoiceCallDefinition=le.fit_transform(dataOne["语音通话清晰度"])
VoiceCallStability=le.fit_transform(dataOne["语音通话稳定性"])

FourFiveUser=le.fit_transform(dataOne["4\\5G用户"])
TerminalBrand=le.fit_transform(dataOne["终端品牌"])

dataOne["语音通话整体满意度"]=pd.DataFrame(OverallSatisfactionVoiceCalls)
dataOne["网络覆盖与信号强度"]=pd.DataFrame(NetworkCoverageSignalStrength)
dataOne["语音通话清晰度"]=pd.DataFrame(VoiceCallDefinition)
dataOne["语音通话稳定性"]=pd.DataFrame(VoiceCallStability)

dataOne["4\\5G用户"]=pd.DataFrame(FourFiveUser)
dataOne["终端品牌"]=pd.DataFrame(TerminalBrand)
dataOne


# In[22]:


def complain(x):
    if x!=0:
        return 1
    else:
        return 0


for i in range(len(dataOne)):
    dataOne.loc[i, '是否投诉']=complain(dataOne.loc[i, '是否投诉'])

dataOne


# In[23]:


dataOne['是否5G网络客户'] = dataOne['是否5G网络客户'].astype('int64')
dataOne['客户星级标识'] = dataOne['客户星级标识'].astype('int64')
dataOne


# In[24]:


dataOne.describe()


# ## 用户行为分析

# In[25]:


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

box_data = dataOne[['语音通话整体满意度',
                    '网络覆盖与信号强度',
                    '语音通话清晰度',
                    '语音通话稳定性',]]
plt.grid(True)
plt.boxplot(box_data,
            notch = True,
            sym = "b+",
            vert = False,
            showmeans = True,
            labels = ['语音通话整体满意度',
                      '网络覆盖与信号强度',
                      '语音通话清晰度',
                      '语音通话稳定性',])
plt.yticks(size=14)
plt.xticks(size=14, font='Times New Roman')
plt.tight_layout()
plt.savefig('figuresOne\\[附件1][语音通话整体满意度、网络覆盖与信号强度、语音通话清晰度、语音通话稳定性]评分箱线图.pdf')


# In[26]:


sns.pairplot(dataOne[['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度','语音通话稳定性']],kind='scatter',diag_kind='kde')
plt.savefig('figuresOne\\[附件1][语音通话整体满意度、网络覆盖与信号强度、语音通话清晰度、语音通话稳定性]评分联合分布图.pdf',bbox_inches='tight')


# ## 划分高分组和低分组

# In[27]:


dataOne['场所合计']=dataOne.loc[:,['居民小区','办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明']].apply(lambda x1:x1.sum(),axis=1)
dataOne['出现问题合计']=dataOne.loc[:,['手机没有信号','有信号无法拨通','通话过程中突然中断','通话中有杂音、听不清、断断续续','串线','通话过程中一方听不见','其他，请注明.1']].apply(lambda x1:x1.sum(),axis=1)
dataOne['脱网次数、mos质差次数、未接通掉话次数合计']=dataOne.loc[:,['脱网次数','mos质差次数','未接通掉话次数']].apply(lambda x1:x1.sum(),axis=1)
dataOne['整体评分']=dataOne.loc[:,['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度','语音通话稳定性']].apply(lambda x1:round(x1.mean()),axis=1)
dataOne


# In[28]:


dataOneHigh = dataOne[(dataOne['语音通话整体满意度']>=7)&(dataOne['网络覆盖与信号强度']>=7)&(dataOne['语音通话清晰度']>=7)&(dataOne['语音通话稳定性']>=7)]
dataOneLow = dataOne[(dataOne['语音通话整体满意度']<=4)&(dataOne['网络覆盖与信号强度']<=4)&(dataOne['语音通话清晰度']<=4)&(dataOne['语音通话稳定性']<=4)]


# In[29]:


dataOneHigh.describe()


# In[30]:


dataOneLow.describe()


# ## 特征分析

# In[31]:


sns.pairplot(dataOneHigh[['整体评分','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']],kind='scatter',diag_kind='kde')
plt.savefig('figuresOne\\[附件1]高分组[场所合计、出现问题合计、脱网次数、mos质差次数、未接通掉话次数合计]评分多变量联合分布图.pdf',bbox_inches='tight')


# In[32]:


sns.pairplot(dataOneLow[['整体评分','场所合计','出现问题合计','脱网次数、mos质差次数、未接通掉话次数合计']],kind='scatter',diag_kind='kde')
plt.savefig('figuresOne\\[附件1]低分组[场所合计、出现问题合计、脱网次数、mos质差次数、未接通掉话次数合计]评分多变量联合分布图.pdf',bbox_inches='tight')


# In[33]:


sns.jointplot(x='出现问题合计', y='整体评分', data=dataOneHigh, kind='hex')
plt.savefig('figuresOne\\[附件1]高分组出现问题合计分布情况.pdf',bbox_inches='tight')


# In[34]:


sns.jointplot(x='出现问题合计', y='整体评分', data=dataOneLow, kind='hex',color='r')
plt.savefig('figuresOne\\[附件1]低分组出现问题合计分布情况.pdf',bbox_inches='tight')


# In[35]:


sns.jointplot(x='场所合计',y='整体评分',data=dataOneHigh,kind='hex')
plt.savefig('figuresOne\\[附件1]高分组场所合计分布情况.pdf',bbox_inches='tight')


# In[36]:


sns.jointplot(x='场所合计',y='整体评分',data=dataOneLow,kind='hex',color='r')
plt.savefig('figuresOne\\[附件1]低分组场所合计分布情况.pdf',bbox_inches='tight')


# In[37]:


sns.jointplot(x='脱网次数、mos质差次数、未接通掉话次数合计',y='整体评分',data=dataOneHigh,kind='hex')
plt.savefig('figuresOne\\[附件1]高分组脱网次数、mos质差次数、未接通掉话次数合计分布情况.pdf',bbox_inches='tight')


# In[38]:


sns.jointplot(x='脱网次数、mos质差次数、未接通掉话次数合计',y='整体评分',data=dataOneLow,kind='hex',color='r')
plt.savefig('figuresOne\\[附件1]低分组脱网次数、mos质差次数、未接通掉话次数合计分布情况.pdf',bbox_inches='tight')


# In[39]:


dataOneHigh['终端品牌'].mode()


# In[40]:


dataOneLow['终端品牌'].mode()


# ## 异常用户评分数据剔除

# In[41]:


dataOneSample=dataOne[((dataOne['其他，请注明']==1)|(dataOne['其他，请注明.1']==1))|((abs(dataOne['语音通话整体满意度']-dataOne['网络覆盖与信号强度'])<=5)&(abs(dataOne['语音通话整体满意度']-dataOne['语音通话清晰度'])<=4)&(abs(dataOne['语音通话整体满意度']-dataOne['语音通话稳定性'])<=4)&(dataOne['网络覆盖与信号强度']-dataOne['语音通话清晰度']<=4)&(dataOne['网络覆盖与信号强度']-dataOne['语音通话稳定性']<=4)&(dataOne['语音通话清晰度']-dataOne['语音通话稳定性']<=3))]
dataOneSample


# In[42]:


dataOne


# In[43]:


sns.heatmap(dataOne[['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度','语音通话稳定性']].corr(method='pearson'),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
plt.title('语音业务评分皮尔逊相关系数热力图')
plt.savefig('figuresOne\\[附件1]语音业务评分皮尔逊相关系数热力图.pdf',bbox_inches='tight')

