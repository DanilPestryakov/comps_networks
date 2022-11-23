import time
from threading import Thread

from routers import Router, DesignatedRouter

designed_router: DesignatedRouter = None

stop_flag = False
printer_flag = False
blink_conn_arr = []


def router_run(neighbors):
    global designed_router
    global blink_conn_arr
    global printer_flag

    conn, index = designed_router.add_connection()
    router = Router(conn, index)
    router.neighbors = neighbors.copy()
    router.router_start()

    while True:
        router.proc_message()
        if blink_conn_arr[router.index]:
            router.router_off()
            time.sleep(5)
            router.router_start()
            blink_conn_arr[router.index] = False
            time.sleep(1)
            printer_flag = True

        if stop_flag:
            break


def designed_router_run():
    global designed_router
    global printer_flag
    designed_router = DesignatedRouter()

    while True:
        designed_router.proc_message()
        if printer_flag:
            designed_router.print_shortest_ways()
            printer_flag = False
        if stop_flag:
            break


def stopper():
    global stop_flag
    time.sleep(10)
    stop_flag = True


def printer():
    global printer_flag
    while True:
        time.sleep(1)

        if stop_flag:
            break


def connections_breaker():
    global printer_flag
    global blink_conn_arr
    time.sleep(3)
    blink_conn_arr[2] = True
    time.sleep(0.3)
    printer_flag = True
    time.sleep(3)
    blink_conn_arr[4] = True
    time.sleep(0.3)
    printer_flag = True


def simulate(nodes, neighbors):
    global printer_flag
    global blink_conn_arr

    dr_thread = Thread(target=designed_router_run, args=())

    node_threads = [Thread(target=router_run, args=(neighbors[i],)) for i in range(len(nodes))]
    blink_conn_arr = [False for i in range(len(nodes))]

    dr_thread.start()
    for i in range(len(nodes)):
        time.sleep(0.2)
        node_threads[i].start()

    conn_breaker_thread = Thread(target=connections_breaker, args=())
    conn_breaker_thread.start()
    time.sleep(0.5)
    printer_flag = True

    time.sleep(10)
    global stop_flag
    stop_flag = True
    for i in range(len(nodes)):
        node_threads[i].join()

    dr_thread.join()


def main():
    linear = {
        "nodes": [0, 1, 2, 3, 4],
        "neighbors": [[1], [0, 2], [1, 3], [2, 4], [3]]
    }
    circle = {
        "nodes": [0, 1, 2, 3, 4],
        "neighbors": [[4, 1], [0, 2], [1, 3], [2, 4], [3, 0]]
    }
    star = {
        "nodes": [0, 1, 2, 3, 4],
        "neighbors": [[2], [2], [0, 1, 3, 4], [2], [2]]
    }

    cur_topology = star
    simulate(cur_topology["nodes"], cur_topology["neighbors"])

    pass


if __name__ == '__main__':
    main()
