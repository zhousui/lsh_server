# -*- coding: utf8 -*-
# ! /usr/bin/python
import numpy as np

import os
import pickle

import zmq
import time
import msgpack
import msgpack_numpy as m

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
        # if angle_x > 30 or angle_y > 30:
        #     return None,None
        t1=time.time()
        #serialized = pickle.dumps(emb_array, protocol=0)
        serialized=msgpack.packb(emb_array, default=m.encode)
        t2=time.time()
        print ("1111:%f"%(t2-t1))
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
                    print ("222:%f"%(t2-t1))
                    print deserialized
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

f=np.load('50w.npy')
l=np.load('labels_50w.npy')
ie=Identifier('127.0.0.1','5555')
# t1=time.time()
# names,pred=ie.identifys(f[1000])
# t2=time.time()
# print t2-t1

total=100
accuracy = 0
error=0
st=time.time()
for i in range(total):
    s=np.random.randint(0,f.shape[0])
    t1=time.time();r=ie.search(f[s].reshape(1,128),[],[]);t2=time.time()
    print ("searching No. %d,query %d,result:[%s],\ncost time:%f"%(i,s,",".join([str(x) for x in r]),t2-t1))
    # if s == r[0]:
    #     accuracy1+=1
    if l[s] in r[0]:
        accuracy+=1
    else:
        error+=1
        print ("FALSE ------ query index %d,return [%s]"%(s,",".join([str(x) for x in r])))
et=time.time()
print ("Total:%d,total search time:%f,cost mean time:%f"%(total,et-st,(et-st)/total))
print ("False count:%d,nearest_neighbor Accuracy rate: %f" %
(error,accuracy / total))
del ie
# print names
# print pred

