# from src.utils.spin_lock import SpinLock
from torch.multiprocessing import Manager
from multiprocessing import RLock
import torch
from jaxtyping import Int64
from torch import Tensor

class StepTracker:
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self):
        # self.lock = SpinLock()
        self.lock = Manager().RLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()

    def set_step(self, step: int) -> None:
        with self.lock:
            self.step.fill_(step)

    def get_step(self) -> int:
        with self.lock:
            return self.step.item()
