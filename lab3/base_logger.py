import enum


class StatsEnum(enum.Enum):
    msg_derived = 'derived'
    msg_lost = 'lost'
    msg_popped = 'popped'


class Logger:
    def __init__(self, name):
        self.name = name
        self.stats = {}
        self.clear_stats()

    def clear_stats(self):
        self.stats = {stat: 0 for stat in StatsEnum}

    def channel_pop(self, channel, need_print=True):
        package = channel.pop()
        self.stats[StatsEnum.msg_popped] += 1
        if need_print:
            print('{} pop {}'.format(self.name, package))
        return package

    def channel_append(self, channel, package, need_print=True):
        result = channel.append(package)
        result = StatsEnum.msg_derived if result else StatsEnum.msg_lost
        self.stats[result] += 1
        if need_print:
            print('{} append {} ({})'.format(self.name, package, result.value))
