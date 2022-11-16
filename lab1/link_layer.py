from multiprocessing.sharedctypes import Synchronized, SynchronizedString

import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px

from link_layer import Connection, ProtocolT, RECEIVER, SENDER
from multiprocessing import Array, Process, Value
from time import time
from typing import List, Tuple


def sender(message_to_send: str, window_size: int, connection: Connection, protocol: ProtocolT, k: Synchronized):
    function = SENDER[protocol]
    bytes_to_send = pickle.dumps(message_to_send)
    sent_bytes = function(bytes_to_send, window_size, connection)
    k.value = len(bytes_to_send) / sent_bytes


def receiver(transmission_error_probability: float, connection: Connection, protocol: ProtocolT,
             received_message: SynchronizedString):
    function = RECEIVER[protocol]
    received_bytes = function(transmission_error_probability, connection)
    received_message.value = pickle.loads(received_bytes).encode()


def run(message_to_send: str, window_size: int, transmission_error_probability: float,
        protocol: ProtocolT) -> Tuple[float, float]:
    connection = Connection()
    k = Value('d')
    received_message = Array('c', len(message_to_send))

    sender_process = Process(target=sender, args=(message_to_send, window_size, connection, protocol, k))
    sender_process.daemon = True
    receiver_process = Process(target=receiver,
                               args=(transmission_error_probability, connection, protocol, received_message))
    receiver_process.daemon = True

    start = time()
    sender_process.start()
    receiver_process.start()
    sender_process.join()
    receiver_process.join()
    end = time()

    print(f'Message to send and received message are equal: {message_to_send == received_message.value.decode()}')

    return k.value, end - start


def const_transmission_error_probability_test(message_to_send: str, transmission_error_probability: int = 0.25,
                                              protocols: List[ProtocolT] = ['go_back_n', 'selective_repeat'],
                                              tests_count: int = 4):
    window_sizes = list(range(1, 16))
    df_ws = pd.DataFrame(columns=['protocol', 'window_size', 'k', 'elapsed_time'])
    base_path = os.path.join('link_layer')

    for protocol in protocols:
        for window_size in window_sizes:
            ks, elapsed_times = [], []
            for _ in range(tests_count):
                k, elapsed_time = run(message_to_send, window_size, transmission_error_probability, protocol)
                ks.append(k)
                elapsed_times.append(elapsed_time)
            df_ws = df_ws.append(dict(protocol=protocol, window_size=window_size, k=np.mean(ks),
                                      elapsed_time=np.mean(elapsed_times)), ignore_index=True)

    fig_ws_k = px.line(df_ws[['protocol', 'window_size', 'k']], x='window_size', y='k', color='protocol')
    fig_ws_k.write_html(os.path.join(base_path, 'window_size_k.html'))

    fig_ws_elapsed_time = px.line(df_ws[['protocol', 'window_size', 'elapsed_time']], x='window_size', y='elapsed_time',
                                  color='protocol')
    fig_ws_elapsed_time.write_html(os.path.join(base_path, 'window_size_elapsed_time.html'))


def const_window_size_test(message_to_send: str, window_size: int = 4, tests_count: int = 4,
                           protocols: List[ProtocolT] = ['go_back_n', 'selective_repeat']):
    transmission_error_probabilities = np.linspace(0, 0.9, 19)
    df_tep = pd.DataFrame(columns=['protocol', 'transmission_error_probability', 'k', 'elapsed_time'])
    base_path = os.path.join('link_layer')

    for protocol in protocols:
        for transmission_error_probability in transmission_error_probabilities:
            ks, elapsed_times = [], []
            for _ in range(tests_count):
                k, elapsed_time = run(message_to_send, window_size, transmission_error_probability, protocol)
                ks.append(k)
                elapsed_times.append(elapsed_time)
            df_tep = df_tep.append(dict(protocol=protocol, k=np.mean(ks), elapsed_time=np.mean(elapsed_times),
                                        transmission_error_probability=transmission_error_probability),
                                   ignore_index=True)

    fig_tep_k = px.line(df_tep[['protocol', 'transmission_error_probability', 'k']], x='transmission_error_probability',
                        y='k', color='protocol')
    fig_tep_k.write_html(os.path.join(base_path, 'transmission_error_probability_k.html'))

    fig_tep_elapsed_time = px.line(df_tep[['protocol', 'transmission_error_probability', 'elapsed_time']],
                                   x='transmission_error_probability', y='elapsed_time', color='protocol')
    fig_tep_elapsed_time.write_html(os.path.join(base_path, 'transmission_error_probability_elapsed_time.html'))


def main():
    test_message = 'Message for testing go back n and selective repeat methods. Lab 1 computer networks..'
    const_transmission_error_probability_test(test_message)
    #const_window_size_test(test_message)


if __name__ == '__main__':
    main()
