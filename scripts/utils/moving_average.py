
class MovingAverage:

    def __init__(self, horizon=3):
        self.buffer = []
        self.horizon = horizon
    
    def __call__(self, new_value):
        if len(self.buffer) == self.horizon:
            self.buffer.pop(0)
        self.buffer.append(new_value)

        return sum(self.buffer)/len(self.buffer)