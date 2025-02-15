from functools import reduce


class MultiOptimizer:
    def __init__(self, optimizers={}):
        self.optimizers = optimizers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x, y: x + y, [value.param_groups for value in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict() for key in self.keys)]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, value in state_dict:
            try:
                self.optimizers[key].load_state_dict(value)
            except:
                print("Unloaded %s :" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler = None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key = None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]


