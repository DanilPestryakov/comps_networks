from threading import Thread, Lock, Barrier
from collections import Counter
import time

from SRP import Sender, Receiver
from channel import Channel


class MsgChannel:
    def __init__(self):
        self._sender = Sender(window_size=4, timeout=1)
        self._receiver = Receiver()
        self._channel_main = Channel()
        self._channel_back = Channel()
        self._sender_thread = None
        self._receiver_thread = None
        self._msg_flag = False
        self.mutex = Lock()

    def push(self, data=None):
        self.mutex.acquire()
        self._receiver.clear_data()
        self._channel_main.clear()
        self._channel_back.clear()
        self._sender_thread = Thread(target=self._sender.run,
                                     args=(self._channel_back, self._channel_main, (data,), False))
        self._receiver_thread = Thread(target=self._receiver.run,
                                       args=(self._channel_main, self._channel_back, False))
        self._sender_thread.start()
        self._receiver_thread.start()
        self._sender_thread.join()
        self._receiver_thread.join()
        self._msg_flag = True
        self.mutex.release()

    def get(self):
        while True:
            self.mutex.acquire()
            if self._msg_flag:
                self._msg_flag = False
                data = self._receiver.data[0] if self._receiver.data else None
                self._receiver.clear_data()
                self.mutex.release()
                break
            self.mutex.release()
            time.sleep(0.1)
        return data


class NetworkNode:
    def __init__(self, index, barrier):
        self.index = index
        self.barrier = barrier
        self._in_connections = {}
        self._out_connections = {}

    def connect(self, other):
        self._in_connections[other.index] = MsgChannel()
        other.add_out_conn(self.index, self._in_connections[other.index])
        self._out_connections[other.index] = MsgChannel()
        other.add_in_conn(self.index, self._out_connections[other.index])

    def add_in_conn(self, index, channel):
        self._in_connections[index] = channel

    def add_out_conn(self, index, channel):
        self._out_connections[index] = channel

    def sendall(self, values):
        pairs = zip(self._out_connections.values(), values)
        sender_threads = [Thread(target=c.push, args=(v,)) for c, v in pairs]
        for thread in sender_threads:
            thread.start()
        for thread in sender_threads:
            thread.join()
        return {i: c.get() for i, c in self._in_connections.items()}


class General(NetworkNode):
    def __init__(self, index, barrier, byzantine=False):
        super(General, self).__init__(index, barrier)
        self.byzantine = byzantine

    def run(self):
        if self.byzantine:
            pattern = 'f' + str(self.index) + '_{}'
            values = [pattern.format(i) for i in self._out_connections.keys()]
        else:
            values = ['t' + str(self.index) for i in self._out_connections.keys()]
        t1_results = self.sendall(values)
        self.barrier.wait()
        self.barrier.reset()
        print('General{} got: {}'.format(self.index, t1_results))
        if self.byzantine:
            pattern = 'f' + str(self.index) + '_{}{}'
            values = []
            for i in self._out_connections.keys():
                column = t1_results.copy()
                for j in column.keys():
                    column[j] = pattern.format(i, j)
                values.append(column)
        else:
            values = [t1_results for i in self._out_connections.keys()]
        t2_results = self.sendall(values)
        print('General{} got: {}'.format(self.index, t2_results))
        self.barrier.wait()
        info_matrix = list(t2_results.values())
        info_matrix.append(t1_results)
        indices = set()
        for column in info_matrix:
            for index in column.keys():
                indices.add(index)
        result = {}
        for index in indices:
            counter = Counter()
            for column in info_matrix:
                if index in column.keys():
                    counter[column[index]] += 1
            common = counter.most_common(2)
            if len(common) > 1 and common[0][1] == common[1][1]:
                result[index] = None
            else:
                result[index] = common[0][0]
        keys = sorted(list(result.keys()))
        result_sorted = {k: result[k] for k in keys}
        print('General{} result: {}'.format(self.index, result_sorted))


if __name__ == "__main__":
    n_generals = 3
    barrier = Barrier(n_generals)
    generals = [General(i, barrier) for i in range(n_generals - 1)]
    generals.append(General(n_generals - 1, barrier, True))
    for g1 in generals:
        for g2 in generals:
            if g1 != g2:
                g1.connect(g2)
    threads = [Thread(target=g.run) for g in generals]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
