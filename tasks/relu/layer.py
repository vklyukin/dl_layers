import numpy as np

from ..utils import Module


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        # self.output = ...
        return self.output

    def updateGradInput(self, input, gradOutput):
        # self.gradInput = ...
        return self.gradInput

    def __repr__(self):
        return "ReLU"
