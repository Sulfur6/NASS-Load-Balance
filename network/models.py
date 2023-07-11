from typing import List


class Forwarder:
    """
    class Forwarder
    """
    def __init__(self, idx, capacity):
        self.idx = idx
        self.capacity = capacity
        self.cap_remain = capacity

    def __lt__(self, other):
        return self.cap_remain > other.cap_remain


class Network:
    def __init__(self, fw_set: List[Forwarder]):
        self.forwarder_count = len(fw_set)
        self.forwarder_set = fw_set
