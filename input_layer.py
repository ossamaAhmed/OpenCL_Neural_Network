import numpy as np
from opencl import OpenCl

class InputLayer(object):

    def __init__(self, inputs, opencl_context):
        self.inputs = opencl_context.create_buffer_input(inputs)
        self.inputs_shape = np.shape(inputs)
        self.num_of_neurons = self.inputs_shape[1]

    def forward_propagation(self):
        self.output = self.inputs #inputs (1,num_of_neurons)
    
    def get_output(self):
        return self.output

    def backward_propagation(self, targets):
        return
