# -*- coding: utf8 -*-
# ! /usr/bin/python
from __future__ import division
import numpy as np
import falconn as fc
import time
import collections

DISTANCE_THRESHOLD=1.0
class Identifier:
    def __init__(self,feature_file,label_file,id_feature_file,id_label_file):
        self.idfeature=np.load(id_feature_file)
        self.idlabel=np.load(id_label_file)

        self.label = np.load(label_file)
        print "start load feature data"
        t1 = time.time()
        feature = np.load(feature_file)
        t2 = time.time()
        print ("load cost time:%f" % (t2 - t1))
        dp = fc.get_default_parameters(feature.shape[0], feature.shape[1],
                                       fc.DistanceFunction.EuclideanSquared)
        ds = fc.LSHIndex(dp)
        train_st = time.time()
        ds.setup(feature)
        train_et = time.time()
        print ("train cost time:%f" % (train_et - train_st))
        self.qo = ds.construct_query_object()

    def construct_name_confidence(self,array):
        names=[self.label[i] for i in array]
        freq = collections.Counter(names)
        s=sum(freq.values())
        for k,v in freq.items():
            freq[k]=v/s
        return freq.keys(),freq.values()

    def compute_norm(self,f,a_query):
        dic={}
        for j in range(len(a_query)):
            q=a_query[j]
            dis_min = DISTANCE_THRESHOLD
            index=0
            for i in range(f.shape[0]):
                dis=np.linalg.norm(f[i]-q)
                if dis < dis_min:
                    dis_min=dis
                    index=i
            if dis_min < DISTANCE_THRESHOLD:
                dic[j]=(index,dis_min)
        dis_min=DISTANCE_THRESHOLD
        #min_v=None
        for k,v in dic.items():
            if dic[k][1]<dis_min:
                dis_min=v[1]
                min_index=v[0]
                #min_v=a_query[k]

        return k,min_index,dis_min,dic

    def identifys(self, a_query):
        k, min_index, dis_min, dic=self.compute_norm(self.idfeature,a_array)
        if min_index == None:
            print "result: Null"
            return [],[]
        t1 = time.time()
        r = self.qo.find_k_nearest_neighbors(idfeature[min_index], 5)
        print r
        t2 = time.time()
        print ("\nresult:[%s],cost time:%f" % (",".join([str(x) for x in r]), t2 - t1))
        return self.construct_name_confidence(r)

#f=np.load('50w.npy')
error = np.load('error/error.npy')
ie=Identifier('error/features.npy','error/labels.npy')
print ie.identifys(f[1000])