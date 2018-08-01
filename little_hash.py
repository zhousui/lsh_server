# -*- coding=utf-8 -*-
import numpy as np


EdisLimit = 1000.0
cosDisLimit = 0.0
disRes = []
cosDisRes = []

def cal_e_dis(feature1,feature2):
    # 计算特征欧式距离
    dis = np.linalg.norm(feature1 - feature2, 2)
    return dis
	
def cal_cos_dis(fea1,fea2):
    # 计算余弦距离
    cos_angle = ( np.dot(fea1, fea2) / ((np.linalg.norm(fea1, 2) * np.linalg.norm(fea2, 2))) + 1) / 2
    return cos_angle

def find_nearest_neighbor(f,v):
    dis_min = 1000.0#EdisLimit
    for i in range(f.shape[0]):
        dis=np.linalg.norm(f[i]-v)
        if dis < dis_min:
            dis_min=dis
            index=i
    return i


def calc_accuracy(dataAll,labelsAll,validData,ValidLabels,NumLimit = 1000):
    # 计算查询准确度,由两部分决定:特征提取准确度,hash准确度
    global EdisLimit
    global cosDisLimit
    global disRes
    global cosDisRes
    print('validData length:',validData.shape[0])
    count = 0
    ecount = 0
    accuracy = 0
    miss = 0
    if validData.shape[0]>NumLimit:
        ind = np.random.randint(0,validData.shape[0],NumLimit)
    else:
        ind = range(validData.shape[0])
    for i in ind:
        queryFeature = validData[i]
        res_ind = find_nearest_neighbor(dataAll,queryFeature)
        fdis = cal_e_dis(dataAll[res_ind],queryFeature)
        disRes.append(fdis)
        cosDis = cal_cos_dis(dataAll[res_ind],queryFeature)
        cosDisRes.append(cosDis)
        if fdis<=EdisLimit and cosDis>cosDisLimit:
            if ValidLabels[i]==labelsAll[res_ind]:
                count += 1.0
            else:
                ecount += 1.0
        if labelsAll[res_ind][0] != ValidLabels[i][0]:
            miss += 1
    accuracy = count/max(1.,min(NumLimit,validData.shape[0]))
    Eaccuracy = ecount/max(1.,min(NumLimit,validData.shape[0]))
    missrate = miss/max(1.,min(NumLimit,validData.shape[0]))
    return accuracy,Eaccuracy,missrate

T98PeopleIDdata=np.load('98人_特征/证件照/signatures.npy')
T98PeopleIDlabel=np.load('98人_特征/证件照/labels_name.npy')
T98PeopleCameradata=np.load('98人_特征/摄像头/signatures.npy')
T98PeopleCameralabel=np.load('98人_特征/摄像头/labels_name.npy')

accuracy,errProb,missrate=calc_accuracy(T98PeopleIDdata,T98PeopleIDlabel,T98PeopleCameradata,T98PeopleCameralabel)

print('Hash accuracy:',accuracy)
print('Hash err     :',errProb)
print('Miss rate     :',missrate)