import time
from multiprocessing import Value

class SpinLock:
    def __init__(self, max_attempts: int = 1024):
        # int type shared variable : 0 - unlocked / 1 - locked
        self.flag = Value('i', 0)
        self.max_attempts = max_attempts

    def acquire(self):
        for attempt in range(self.max_attempts):
            # Access flag atomically using get_lock()
            with self.flag.get_lock():
                if self.flag.value == 0:
                    self.flag.value == 1
                    return
            # Sleep to prevent overusing CPU
            time.sleep(0)
        raise TimeoutError(f"Failed to acquire spin lock after {self.max_attempts} attempts")

    def release(self):
        with self.flag.get_lock():
            self.flag.value = 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()