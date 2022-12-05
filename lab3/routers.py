import enum

import topology as topology_class


class Router:

    def __init__(self, conn, index):
        self.DR_connection = conn
        self.topology = topology_class.Topology()
        self.shortest_roads = None
        self.index = index
        self.neighbors = []

    def print_shortest_ways(self):
        shortest_ways = self.topology.get_shortest_ways(self.index)
        result = 'shortest ways from {}:\n'.format(self.index)
        for i, way in enumerate(shortest_ways):
            if way:
                way_str = ' -> '.join(map(str, way))
            else:
                way_str = None
            result += '({}, {}): {}\n'.format(self.index, i, way_str)
        print(result)

    def send_neighbors(self):
        msg = Message()
        msg.type = MsgType.NEIGHBORS
        msg.data = self.neighbors.copy()
        self.DR_connection.send_message(msg)

    def get_topology(self):
        msg = Message()
        msg.type = MsgType.GET_TOPOLOGY
        self.DR_connection.send_message(msg)

    def router_start(self):
        self.send_neighbors()
        self.get_topology()

    def router_off(self):
        msg = Message()
        msg.type = MsgType.OFF
        self.DR_connection.send_message(msg)

    def add_node(self, index, neighbors):
        # print(f".{index}.")
        self.topology.add_new_node(index)
        for j in neighbors:
            self.topology.add_new_link(index, j)

        if index in self.neighbors:
            if index not in self.topology.topology[self.index]:
                msg = Message()
                msg.type = MsgType.NEIGHBORS
                msg.data = [index]
                self.DR_connection.send_message(msg)

    def delete_node(self, index):
        self.topology.delete_node(index)

    def proc_message(self):
        input_msg = self.DR_connection.get_message()

        if input_msg is None:
            return

        text = 'router' + str(self.index) + ' got: "{}"'
        if input_msg.type == MsgType.NEIGHBORS:
            index = input_msg.data["index"]
            neighbors = input_msg.data["neighbors"]
            ways = ['{}->{}'.format(index, i) for i in neighbors]
            print(text.format('LSU ' + ', '.join(ways)) + '\n')
            self.add_node(index, neighbors)

        elif input_msg.type == MsgType.SET_TOPOLOGY:
            new_topology = input_msg.data
            self.topology = new_topology

        elif input_msg.type == MsgType.OFF:
            index = input_msg.data
            print(text.format('LSU {} dropped'.format(index)) + '\n')
            self.delete_node(index)

        elif input_msg.type == MsgType.PRINT_WAYS:
            self.print_shortest_ways()

        else:
            print("DR: unexpected msf type:", input_msg.type)

    pass


class DesignatedRouter:

    def __init__(self):
        self.connections = []
        self.topology = topology_class.Topology()

    def add_connection(self):
        new_connection = Connection()
        new_index = len(self.connections)
        self.connections.append(new_connection)
        return new_connection, new_index

    def add_node(self, index, neighbors):
        self.topology.add_new_node(index)
        for j in neighbors:
            self.topology.add_new_link(index, j)

    def delete_node(self, index):
        self.topology.delete_node(index)

    def send_all_exclude_one(self, exclude_index, msg):
        for conn_ind in range(len(self.connections)):
            conn = self.connections[conn_ind]
            if conn is None:
                continue
            if conn_ind == exclude_index:
                continue
            conn.send_message(msg, 1)

    def proc_msg_neighbors(self, conn_ind, input_msg):
        self.add_node(conn_ind, input_msg.data)

        msg = Message()
        msg.type = MsgType.NEIGHBORS
        msg.data = {"index": conn_ind,
                    "neighbors": input_msg.data
                    }

        self.send_all_exclude_one(conn_ind, msg)

    def proc_msg_off(self, conn_ind, input_msg):
        self.delete_node(conn_ind)

        msg = Message()
        msg.type = MsgType.OFF
        msg.data = conn_ind

        self.send_all_exclude_one(conn_ind, msg)

    def print_shortest_ways(self):
        msg = Message()
        msg.type = MsgType.PRINT_WAYS
        for conn in self.connections:
            conn.send_message(msg, 1)

    def proc_message(self):
        for conn_ind in range(len(self.connections)):
            conn = self.connections[conn_ind]
            if conn is None:
                continue

            input_msg = conn.get_message(1)

            if input_msg is None:
                continue

            # print(f"dr({conn_ind}): {input_msg}\n", end="")

            if input_msg.type == MsgType.NEIGHBORS:
                self.proc_msg_neighbors(conn_ind, input_msg)

            elif input_msg.type == MsgType.GET_TOPOLOGY:
                msg = Message()
                msg.type = MsgType.SET_TOPOLOGY
                msg.data = self.topology.copy()
                conn.send_message(msg, 1)

            elif input_msg.type == MsgType.OFF:
                self.proc_msg_off(conn_ind, input_msg)

            else:
                print("DR: unexpected msf type:", input_msg.type)


class MsgType(enum.Enum):
    NEIGHBORS = enum.auto()
    GET_TOPOLOGY = enum.auto()
    SET_TOPOLOGY = enum.auto()
    OFF = enum.auto()
    PRINT_WAYS = enum.auto()


class Message:
    def __init__(self):
        self.data = None
        self.type = None

    def __str__(self):
        return f"({self.type}: {self.data})"


class Connection:

    # owner -> node # right
    # owner <- node # left
    def __init__(self):
        self.right_queue = []
        self.left_queue = []

    def __str__(self):
        return f"(->:{self.right_queue}\n<-:{self.right_queue})"

    @staticmethod
    def __get_message(queue, ):
        if len(queue) > 0:
            result = queue[0]
            queue.pop(0)
            return result
        else:
            return None

    def get_message(self, direction=0):
        if direction == 0:
            res = self.__get_message(self.right_queue)
            if res:
                # print(f"get  ->:{res}\n")
                pass
            return res
        else:
            res = self.__get_message(self.left_queue)
            if res:
                # print(f"get  <-:{res}\n")
                pass
            return res

    def send_message(self, message, direction=0):
        if direction == 0:
            self.left_queue.append(message)
            # print(f"send <-: {message}\n")
            return
        else:
            self.right_queue.append(message)
            # print(f"send ->: {message}\n")
            return
