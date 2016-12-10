import numpy as np

class Layer(object):

    def __init__(self, num_of_neurons, activation_func, learning_rate, regulaization, opencl_context):
	self.opencl_context = open_cl_context
        self.num_of_neurons = num_of_neurons
        self.prev_layer = None
        self.next_layer = None
        self.weights = None
        self.bias = None
        self.output = None
        self.epsilon = learning_rate
        self.reg_lambda = regulaization
        self.activation_function = activation_func

    def link_prev(self, prev_layer):
        self.prev_layer = prev_layer
        self.prev_layer.next_layer = self
        self.weights = np.random.randn(self.prev_layer.num_of_neurons, self.num_of_neurons) / np.sqrt(self.prev_layer.num_of_neurons)
        self.weights_shape = self.weights.shape
	self.weights = opencl_context.create_buffer_input(self.weights)
	self.bias = np.zeros((1, self.num_of_neurons))
	self.bias_shape = self.bias.shape
	self.bias = opencl_context.create_buffer_input(self.bias)
	self.output = np.zeros((1, self.num_of_neurons))
	self.output_shape = self.output.shape
	self.output = opencl_context.create_buffer_output(outputs) 

    def forward_propagation(self):
        self.prev_layer.forward_propagation()
	open_cl_context.calc_dot_product(self.prev_layer.output, self.prev_layer.output_shape
					, self.weights, self.weights_shape, self.output, self.output_shape)
        pre_output = self.prev_layer.output.dot(self.weights) + self.bias
        self.output = self.activation_function(pre_output)
    
    def get_output(self):
        if self.next_layer == None:
            self.probs = self.output / np.sum(self.output, axis=1, keepdims=True)
            return self.probs
        else:
            return self.output
    
    def backward_propagation(self, targets):
        if self.next_layer == None: #this is the output layer
            self.get_output()
            self.layer_delta = self.probs
            self.layer_delta[range(len(targets)), targets] -= 1
            dW = (self.prev_layer.get_output().T).dot(self.layer_delta)
            dB = np.sum(self.layer_delta, axis = 0, keepdims=True)
        else:
            self.layer_delta = self.next_layer.layer_delta.dot(self.next_layer.weights.T)*(1-np.power(self.output, 2))
            dW = np.dot(self.prev_layer.get_output().T, self.layer_delta)
            dB = np.sum(self.layer_delta, axis=0)
        dW += self.reg_lambda * self.weights
        self.weights += -self.epsilon * dW
        self.bias += -self.epsilon * dB
        self.prev_layer.backward_propagation(targets)
            
