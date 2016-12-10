import pyopencl as cl
import numpy
import time
import matplotlib.pyplot as plt


class OpenCl(object):

	def __init__(self, block_size, filename):
		self.block_size = block_size
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		f = open(filename, 'r')
		fstr = "".join(f.readlines())
		self.program = cl.Program(self.ctx, fstr).build()
		self.dot_product = self.program.matrixMul
		self.forward_propagation = self.program.forwardPropagation
		self.mf = cl.mem_flags
	def get_kernel(self, kernel_name):
		return program.kernel_name
	def create_buffer_input(self, inputs):
		return cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=inputs)
	def create_buffer_output(self, outputs):
		return cl.Buffer(self.ctx, seld.mf.READ_ONLY | self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR, outputs.nbytes)
	
	def calc_dot_product(self,matrixa_buff, a_shape, matrixb_buff, b_shape, matrixc_buff, c_shape):
		self.dot_product(self.queue, c_shape, None, matrixc_buff, numpy.int32(a_shape[1])				,numpy.int32(a_shape[0]), numpy.int32(b_shape[1]), 
				numpy.int32(b_shape[0]), matrixa_buff, matrixb_buff)

	def calc_forward_propagation(self,matrixa_buff, a_shape, matrixb_buff, b_shape, matrixc_buff, c_shape, bias,ac):
                self.forward_propagation(self.queue, c_shape, None, matrixc_buff, numpy.int32(a_shape[1])                               ,numpy.int32(a_shape[0]), numpy.int32(b_shape[1]),
                                numpy.int32(b_shape[0]), matrixa_buff, matrixb_buff, numpy.int32(bias), 
				numpy.int32(ac))
	

	def copy_buffer_to_host(self, buff, outputs):
		cl.enqueue_copy(self.queue, buff, outputs).wait()	

block_size = 128
print (cl.device_info.MAX_WORK_GROUP_SIZE)
#ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx)
filename = "kernel.cl"
opencl_context = OpenCl(block_size, filename)
#allocate h mem for A and B
experiments = [256]
num_of_connections = []
kernel_time = []
kernel_time_in = []
non_parallelized = []
copy_back = []
for experiment in experiments:
	w_a = numpy.int32(1024)
	h_a = numpy.int32(256)
	w_b = numpy.int32(experiment)
	h_b = numpy.int32(1024)
	a = numpy.random.randint(30, size=(h_a, w_a)).astype(numpy.float32)
	b = numpy.random.randint(30, size=(h_b, w_b)).astype(numpy.float32)
	c = numpy.random.randint(30, size=(h_a, w_b)).astype(numpy.float32)
	#allocate d mem for A, B and C
	start1 = time.time()
	d_a_buf = opencl_context.create_buffer_input(a)
	d_b_buf = opencl_context.create_buffer_input(b)
	dest_buf = opencl_context.create_buffer_input(c)
	start2 = time.time()	
	opencl_context.calc_dot_product(d_a_buf, a.shape, d_b_buf, b.shape, dest_buf, c.shape)	
	end = time.time()
	num_of_connections.append(w_b*h_b)
	kernel_time.append(end-start2)
	#start = time.time()
	startcopyback = time.time()
	opencl_context.copy_buffer_to_host(dest_buf, c)
	end = time.time()
	copy_back.append(end-startcopyback)
	#print ("c",c)
	kernel_time_in.append(end-start1)
	start = time.time()
	c = a.dot(b)
	#print ("c", c)
	end = time.time()
	non_parallelized.append(end-start)
	#print ("a", a)
#print ("b", b)
#print ("c", c)
print ("num of connections", num_of_connections)
print ("kernel time ", kernel_time)
print ("kernel time in/out ", kernel_time_in)
print ("non_parallelized", non_parallelized)
print ("copying back data time", copy_back)
"""plt.plot(num_of_connections, kernel_time, 'r', label='kernel_time')
plt.plot(num_of_connections, non_parallelized, 'b', label='non parallelized w numpy')
#plt.plot(num_of_connections, non_parallelized, 'g', label='non parallelized w numpy')
#plt.plot(num_of_connections, kernel_time, 'r', label='kernel_time', num_of_connections, kernel_time_in, 'b',label='non_parallelized' )
plt.legend(loc='upper left')
plt.xlabel('number of Connections', fontsize =16)
plt.ylabel('Execution Time (seconds)', fontsize=16)
plt.show()
plt.plot(num_of_connections, kernel_time, 'r', label='kernel_time')
plt.plot(num_of_connections, kernel_time_in, 'g', label='kernel_time w in/out')
plt.plot(num_of_connections, non_parallelized, 'b', label='non parallelized w numpy')
plt.legend(loc='upper left')
plt.xlabel('number of Connections', fontsize =16)
plt.ylabel('Execution Time (seconds)', fontsize=16)
plt.show()"""
