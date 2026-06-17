import numpy

class PixelsFeatureExtractor:

    def __init__(self):
        self.shape = (4, 64, 64)


    def __call__(self, obs):
        return self.process(obs)

    def process(self, obs):
        shifts = numpy.array([3, 2, 1, 0], dtype=numpy.uint8).reshape(4, 1, 1)

        binary_tensor = (obs >> shifts) & 1

        result = numpy.array(binary_tensor, dtype=numpy.float32)

        return result
