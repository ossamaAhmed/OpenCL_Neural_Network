import matplotlib
from layer import Layer
from input_layer import InputLayer
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from opencl import OpenCl
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

def build_neural_network(opencl_context, inputs, targets, layers, learning_rate, regulization):
    input_layer = InputLayer(inputs, opencl_context)
    model_layers = []
    model_layers.append(input_layer)
    for i in range(len(layers)-1):
        layer = Layer(layers[i], np.tanh, learning_rate, regulization, opencl_context)
        layer.link_prev(model_layers[-1])
        model_layers.append(layer)
    #add output layer
    layer = Layer(layers[-1], np.exp, learning_rate, regulization, opencl_context)
    layer.link_prev(model_layers[-1])
    model_layers.append(layer)
    return model_layers

def calculate_loss(model, targets):
    model[-1].forward_propagation()
    prob_output = model[-1].get_output()
    corect_logprobs = -np.log(prob_output[range(len(targets)), targets])
    data_loss = np.sum(corect_logprobs)
    return 1./len(targets) * data_loss

def train_network(model, targets, training_passes):
    for i in range(training_passes):
        model[-1].forward_propagation()
        model[-1].backward_propagation(targets)
        if i % 1000 == 0:
          print ("Loss after iteration %i: %f" %(i, calculate_loss(model, targets)))


block_size = 128
filename = "kernel.cl"
opencl_context = OpenCl(block_size, filename)
"""d_a_buf = opencl_context.create_buffer_input(a)
d_b_buf = opencl_context.create_buffer_input(b)
dest_buf = opencl_context.create_buffer_input(c)
start = time.time()
opencl_context.calc_dot_product(d_a_buf, a.shape, d_b_buf, b.shape, dest_buf, c.shape)
end = time.time()
opencl_context.copy_buffer_to_host(dest_buf, c)
print ("c",c)
print ("time parallelized", end-start)
start = time.time()
"""

model = build_neural_network(opencl_context, X, y, [3, 3, 2], 0.001, 0.01)
train_network(model, y, 2000)


"""layer_one = Layer(30, np.tanh, 0.01, 0.01)
layer_two = Layer(4, np.tanh, 0.01, 0.01)
layer_three = Layer(2, np.exp, 0.01, 0.01)
layer_one.link_prev(input_layer)
layer_two.link_prev(layer_one)
layer_three.link_prev(layer_two)
layer_three.forward_propagation()
print (layer_three.weights)
layer_three.backward_propagation(y)
print (layer_three.weights)"""


