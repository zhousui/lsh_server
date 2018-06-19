# -*- coding: utf8 -*-
# ! /usr/bin/python
from __future__ import division
import numpy as np
import falconn as fc
import time
import threading
import zmq
import sys, getopt
import pickle
import collections
import traceback
import msgpack
import msgpack_numpy as m

class lsh_server:
    def __init__(self,feature_file,label_file,port):
        self.url_worker = 'inproc://ping-workers'
        url_router = "tcp://*:%s"%port
        self.worker_num = 10
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(url_router)
        self.workers = self.context.socket(zmq.DEALER)
        self.workers.bind(self.url_worker)

        self.label = np.load(label_file)
        print "start load feature data"
        t1 = time.time()
        self.feature = np.load(feature_file)
        t2 = time.time()
        print ("load cost time:%f" % (t2 - t1))
        dp = fc.get_default_parameters(self.feature.shape[0], self.feature.shape[1],
                                       fc.DistanceFunction.EuclideanSquared)
        ds = fc.LSHIndex(dp)
        train_st = time.time()
        ds.setup(self.feature)
        train_et = time.time()
        print ("train cost time:%f" % (train_et - train_st))
        self.qo = ds.construct_query_object()


    def construct_name_confidence(self,query_array,indexs):
        names=[]
        confidences=[]
        for i in range(len(indexs)):
            if indexs[i] == None:
                names.append(None)
            else:
                names.append(self.label[indexs[i]])
            cos = np.dot(query_array[i],self.feature[indexs[i]])/(np.linalg.norm(query_array[i])*np.linalg.norm(self.feature[indexs[i]]))
            confidences.append((cos+1)*0.5)
            
        # freq = collections.Counter(names)
        # s=sum(freq.values())
        # for k,v in freq.items():
        #     freq[k]=v/s
        return names,confidences

    def loop(self):
        for i in range(self.worker_num):
            thread = threading.Thread(target=self.worker, args=(i, self.url_worker, self.context,))
            thread.start()
        zmq.device(zmq.QUEUE, self.router, self.workers)
        self.router.close()
        self.workers.close()
        self.context.term()

    def worker(self,name, url_worker, context):
        print('worker {0} start'.format(name))
        worker = context.socket(zmq.REP)
        worker.connect(url_worker)
        while True:
            try:
                message = worker.recv()
                t1 = time.time()
                #query = pickle.loads(message)
                query_arr=msgpack.unpackb(message,object_hook=m.decode)
                # print query_arr
                # print query_arr.dtype
                t2 = time.time()
                print ("unpack cost time:%f"%(t2-t1))
                # print ("this is query:%s"%query)
                # print query
                #indexs=[]
                names=[]
                confidences=[]
                ts=time.time()
                for query in query_arr:
                    result=None
                    t1 = time.time()
                    result = self.qo.find_nearest_neighbor(query)
                    t2 = time.time()
                    print ("\nresult:[%d],cost time:%f"%(result,t2-t1))
                    names.append(self.label[result] if result != None else None)
                    confidences.append(((np.dot(query,self.feature[result])/(np.linalg.norm(query)*np.linalg.norm(self.feature[result])))+1)*0.5 if result != None else None)
                    #indexs.append(result)
                te=time.time()
                print ("Total compute time:%f"%(te-ts))
                #print indexs
                #name_confidence=self.construct_name_confidence(query_arr,indexs)
                name_confidence=(names,confidences)
                t1 = time.time()
                #serialized = pickle.dumps(name_confidence, protocol=0)
                serialized=msgpack.packb(name_confidence,use_bin_type=True)
                t2 = time.time()
                print ("1111:%f"%(t2-t1))
                worker.send(serialized)
                print "after send"
            except Exception, e:
                print 'e.message:\t', e.message
                print 'trace info:\n%s' % traceback.format_exc()
                break
        worker.close()

def usage():
    print "python lsh_server -f feature.npy -l labels.npy -k 5"

def get_arg():
    opts, args = getopt.getopt(sys.argv[1:], "hf:k:l:p")
    feature_file = ""
    label_file = ""
    k = 5
    port = "5555"
    for op, value in opts:
        if op == "-f":
            feature_file = value
        elif op == "-l":
            label_file = value
        elif op == "-k":
            k = int(value)
        elif op == "-p":
            port = value
        elif op == "-h":
            usage()
            sys.exit()
    if feature_file == "" or label_file == "":
            usage()
            sys.exit()

    return feature_file,label_file,k,port


if __name__ == '__main__':
    feature_file,label_file,k,port=get_arg()
    server=lsh_server(feature_file,label_file,port)
    server.loop()