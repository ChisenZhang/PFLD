# !/usr/bin/python
# encoding:utf-8

'''
producer consumer model
'''

import threading
# from threading import Lock
import queue
import numpy as np

def producer(q, dataList, batch_size, processFunc=None):
    globalExpIndex = 0
    while True:
        if globalExpIndex + batch_size > len(dataList):
            globalExpIndex = 0
            print('reset GEXPIndex:', globalExpIndex)
            # random.shuffle(dataList.ids)
            q.put(None)
            return
        if q.full():
            # print('queue is full')
            continue
        imgBatch = []
        lalBatch = []

        tmpL = 0
        for tmp in range(globalExpIndex, globalExpIndex+batch_size):
            if tmp + tmpL >= len(dataList):
                globalExpIndex = 0
                q.put(None)
                return

            while True: # 处理图中框过小，过滤
                tmpImg, tmpLal = dataList[tmp + tmpL] if processFunc is None else processFunc(dataList[tmp+tmpL])
                if tmpImg is None:
                    tmpL += 1
                else:
                    break
            imgBatch.append(tmpImg)
            lalBatch.append(np.array(tmpLal)[:, 0:4].tolist())
            # print(tmpBatch[-1][0].shape, tmpBatch[-1][1].shape)
        # if processFunc is not None:
        #     batch = []
        #     for sample in tmpBatch:
        #         batch.append(processFunc(sample))
        #     tmpBatch = batch
        globalExpIndex += batch_size + tmpL
        print('GEXPIndex:', globalExpIndex)
        q.put([imgBatch, lalBatch])

def consumer(q):
    flag = True
    while True:
        if q.empty():
            if flag:
                continue
            else:
                break
        data = q.get()
        if data is None:
            flag = False
            continue
        print('data:', data)

if __name__ == '__main__':
    q = queue.Queue(2)
    dataList = [[[i], i+1] for i in range(100)]

    prod = threading.Thread(target=producer, args=(q, dataList, 10, ))
    # cons = threading.Thread(target=consumer, args=(q, ))
    prod.start()
    # cons.start()
    flag = True
    while True:
        if q.empty():
            if flag:
                continue
            else:
                break
        data = q.get()
        if data is None:
            flag = False
            continue
        print('data:', data)
    # while not q.empty():
    #     print('data:', q.get())
