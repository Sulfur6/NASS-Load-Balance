from typing import List, Optional
from network.models import Forwarder


class Instance:
    """
    tmp
    """
    def __init__(self, idx, cost):
        self.id = idx
        self.cost = cost
        self.forwarder: Optional[Forwarder] = None

    def __lt__(self, other):
        return self.cost > other.cost


class Tenant:
    """
    tmp
    """
    def __init__(self, ins_list: List[Instance]):
        self.ins_count = len(ins_list)
        self.ins_list = ins_list
