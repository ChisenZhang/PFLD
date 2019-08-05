# !/usr/bin/python
# encoding:utf-8

'''
多进程喂数据
'''

import multiprocessing
import threading
import random
import numpy as np

class DataService(object):
    def __init__(self, source_p, batch_size=32, QMaxLen=32, workers=3):
        self.source_p = source_p
        self.batch_size = batch_size
        self.q = multiprocessing.Queue(QMaxLen)
        self.lock = multiprocessing.Lock()
        self.workers = workers
        self.GIndex = 0

    def start(self):
        print('Starting parallelised augmentation...')
        thread = threading.Thread(target=self.spawn)
        self.is_running = True
        thread.start()

    def stop(self):
        print('Stopping parallelised augmentation...')
        [p.terminate() for p in self.proc_lst]
        self.is_running = False

    def pop(self):
        while True:
            with self.lock:
                if not self.q.empty():
                    return self.q.get()

    def spawn(self):
        print('Spawning', self.workers, 'workers')
        self.proc_lst = []
        for i in range(self.workers):
            p = multiprocessing.Process(target=self.worker)
            self.proc_lst.append(p)
            p.start()
        # while self.is_running:
        #     pass
        [p.join() for p in self.proc_lst]
        self.proc_lst = []

    def worker(self):
        while True:
            try:
                with self.lock:
                    if self.GIndex >= len(self.source_p):
                        random.shuffle(self.source_p.ids)
                        self.GIndex = 0
                imgs, boxes = self.getBatch(self.GIndex)
                with self.lock:
                    if imgs is None:
                        random.shuffle(self.source_p.ids)
                        self.GIndex = 0
                        continue
                    self.GIndex += self.batch_size
                    if not self.q.full():
                        self.q.put(tuple([imgs, boxes]))
            except AssertionError:
                print('Assertion Error (edge-case) - skipping...')

    def getBatch(self, globalExpIndex):
        imgBatch = []
        labBatch = []
        tmpL = 0
        for tmp in range(globalExpIndex, globalExpIndex + self.batch_size):
            if tmp + tmpL >= len(self.source_p):
                return None, None

            while True:  # 处理图中框过小，过滤
                tmpImg, tmpLal = self.source_p[tmp + tmpL]
                if tmpImg is None:
                    tmpL += 1
                else:
                    break
            imgBatch.append(tmpImg)
            labBatch.append(np.array(tmpLal)[:, 0:4].tolist())
        return imgBatch, labBatch