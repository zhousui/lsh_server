# -*- coding: utf8 -*-
# ! /usr/bin/python

'''
#Change log:
#2018-7-18   
#128 dimension feature upgrade to 512 dimension feature
'''
import numpy as np

import os
import pickle

import zmq
import time
import msgpack
import msgpack_numpy as m

SUPPORT_FEATURE_DIMENSION = 128
REQUEST_TIMEOUT = 2500
REQUEST_RETRIES = 3

class Identifier:
    def __init__(self,lshserver,port):
        self.context = zmq.Context()
        self.lshserver=lshserver
        self.port = port
        print "Connecting to lsh server..."
        self.client = self.context.socket(zmq.REQ)
        self.client.connect("tcp://%s:%s"%(lshserver,port))
        print "Connect lsh server success"
        self.poll = zmq.Poller()
        self.poll.register(self.client, zmq.POLLIN)

    def __del__(self):
        self.client.close()
        self.context.term()

    def search(self, emb_array, angle_x_array,angle_y_array):
        if not isinstance(emb_array,np.ndarray):
            raise TypeError('emb_array must be an instance of numpy.ndarray')
        if len(emb_array.shape) != 2:
            raise ValueError('emb_array must be a two-dimensional array')
        #2018-7-18   
        #128 dimension feature upgrade to 512 dimension feature
        # if emb_array.shape[1] != 128:
        if emb_array.shape[1] != SUPPORT_FEATURE_DIMENSION:
            raise ValueError('emb_array dimension mismatch: {} expected, but {} found'.format(
                    SUPPORT_FEATURE_DIMENSION, emb_array.shape[1]))
        if emb_array.dtype != np.float64:
            raise ValueError('data type of emb_array must be np.float64')
        if type(angle_x_array)!=list or type(angle_y_array)!=list:
            raise TypeError('angle_array must be list')
        if len(emb_array) != len(angle_x_array) or len(emb_array) != len(angle_y_array):
            raise ValueError('len of array must be same')
        # if angle_x > 30 or angle_y > 30:
        #     return None,None
        t1=time.time()
        #serialized = pickle.dumps(emb_array, protocol=0)
        serialized=msgpack.packb(emb_array, default=m.encode)
        t2=time.time()
        print ("emb array pack cost time:%f"%(t2-t1))
        retries_left = REQUEST_RETRIES
        names = []
        pred = []

        while retries_left:
            #print "I: Sending (%s)" % emb_array
            self.client.send(serialized)
            expect_reply = True
            while expect_reply:
                socks = dict(self.poll.poll(REQUEST_TIMEOUT))
                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv()
                    if not reply:
                        retries_left -= 1
                        print "E:None reply"
                        #return [],[]
                        break
                    t1=time.time()
                    #deserialized = pickle.loads(reply)
                    deserialized = msgpack.unpackb(reply, raw=False)
                    t2=time.time()
                    print ("msg unpack time:%f"%(t2-t1))
                    #print deserialized
                    if not deserialized:
                        print "E: Malformed reply from server: %s" % reply
                        retries_left -= 1
                    else:
                        #names,pred=deserialized
                        names = deserialized[0]
                        pred = deserialized[1]
                        return deserialized[0],deserialized[1]
                        retries_left = 0
                        expect_reply = False

                else:
                    print "W: No response from server, retrying…"
                    # Socket is confused. Close and remove it.
                    self.client.setsockopt(zmq.LINGER, 0)
                    self.client.close()
                    self.poll.unregister(self.client)
                    retries_left -= 1
                    if retries_left == 0:
                        expect_reply=False
                        print "E: Server seems to be offline, abandoning"
                        break
                    print "I: Reconnecting and resending (%s)" % emb_array
                    # Create new connection
                    self.client = self.context.socket(zmq.REQ)
                    self.client.connect("tcp://%s:%s"%(self.lshserver,self.port))
                    self.poll.register(self.client, zmq.POLLIN)
                    self.client.send(serialized)
        #self.client.close()
        #self.context.term()
        return names,pred


#DELETE FOR FEATURE DIMENSION CHANGE 2018-7-18

# f=np.load('../features.npy')
# l=np.load('../labels.npy')
f=np.load('../datasets/feature_12.npy')
l=np.load('../datasets/labels_12.npy')
ie=Identifier('127.0.0.1','5555')
#ie=Identifier('15.112.148.180','5556')
# t1=time.time()
# names,pred=ie.identifys(f[1000])
# t2=time.time()
# print t2-t1

#ie.search([1,2,3],[1,2,3],[1,3,4])

total=10000
accuracy = 0
error=0
st=time.time()
for i in range(total):
    s=np.random.randint(0,f.shape[0])
    t1=time.time();r=ie.search(f[s].reshape(1,SUPPORT_FEATURE_DIMENSION),[1],[1]);t2=time.time()
    print ("searching No. %d,query %d,query label: %s, result:[%s],\ncost time:%f"%(i,s,l[s],",".join([str(x) for x in r[0]]),t2-t1))
    #print l[s]
    # if s == r[0]:
    #     accuracy1+=1
    if l[s] in r[0]:
        accuracy+=1
    else:
        error+=1
        #print ("FALSE ------ query index %d,return [%s]"%(s,",".join([str(x) for x in r])))
et=time.time()
print ("Total:%d,total search time:%f,Currency num:%f, cost mean time:%f"%(total,et-st,total/(et-st), (et-st)/total))
print ("False count:%d,nearest_neighbor Accuracy rate: %f" %
(error,accuracy / total))
del ie
# print names
# print pred

