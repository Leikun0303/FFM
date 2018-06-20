# 将统计数据按fields处理
# 例如特征banner_pos的统计数据文件为:clickVSbanner_pos.csv
# 其内容为:

# banner_pos	count(click)	sum(click)	avg(click)
# 0	            29109590	    4781901	    0.1642723584
# 1	            11247282	    2065164	    0.1836144946
# 2	            13001	        1550	    0.1192215983
# 3	            2035	        372	        0.1828009828
# 4	            7704	        1428	    0.1853582555
# 5	            5778	        702	        0.1214953271
# 7	            43577	        13949	    0.3201000528

# banner_pos列为该特征在训练集中的所有取值
# count(click)列为每个取值的统计数
# sum(click)列为click=1的统计数
# avg(click)列为点击率,即sum(click)/count(click)

# 对于直接编码的特征,将提取第一列,每个取值归为一个域,生成一个特征字典文件
# 对于按频率编码的特征,将提取第一列和第二列,小于10次的取值归于一个域,

import numpy as np
import pandas as pd
import pickle

direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type']

frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                              'device_model', 'device_id']

train_path='../Input/train'

feature2field = {}
field_index = 0
ind = 0
for field in direct_encoding_fields:
    # value to one-hot-encoding index dict
    field_dict = {}
    # 载入第一列

    if field=='hour':#时间直接给定24小时,因为统计数据的时间取值是和train.csv中一样分布的
        field_sets = pd.DataFrame({'hour':list(range(24))})
    else:
        field_sets=pd.read_csv(train_path+'/features/clickVS'+field+'.csv', usecols=[field])

    for value in list(field_sets[field]):
        field_dict[value] = ind
        feature2field[ind] = field_index
        ind += 1
    field_index += 1
    with open(train_path+'/dicts/'+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


for field in frequency_encoding_fields:
    # value to one-hot-encoding index dict
    field_dict = {}
    #载入第一列和第二列,并组合为一个字典
    field_sets=pd.read_csv(train_path+'/features/clickVS'+field+'.csv', usecols=[field,'count(click)'])
    field2count=dict(zip(list(field_sets[field]), list(field_sets['count(click)'])))
    index_rare = None
    for value,count in field2count.items():
        if count < 10:
            if index_rare == None:
                field_dict[value] = ind
                feature2field[ind] = field_index
                index_rare = ind
                ind += 1
            else:
                field_dict[value] = index_rare
                feature2field[index_rare] = field_index
        else:
            field_dict[value] = ind
            feature2field[ind] = field_index
            ind += 1
    field_index += 1
    with open(train_path+'/dicts/'+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)

# 点击率这列有2个取值
click=[0,1]
field_dict = {}
field_sets = click
for value in list(field_sets):
    field_dict[value] = ind + 1
    ind += 1
with open(train_path+'/dicts/'+'click'+'.pkl', 'wb') as f:
    pickle.dump(field_dict, f)

with open(train_path+'/features/feature2field.pkl', 'wb') as f:
    pickle.dump(feature2field, f)
