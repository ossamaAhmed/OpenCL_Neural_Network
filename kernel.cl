
__kernel void matrixMul(__global float* outputC, int widthA, int heightA, int widthB, int heightB, __global float* inputA, __global float* inputB) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  //int widthA, int heightA, int widthB, int heightB
  //printf("row: %d, col: %d\n", row, column);
  double sum = 0.0;
  for (int i = 0; i < widthA; i++) {
    sum += inputA[row * widthA + i] * inputB[i * widthB + column];
  }
  outputC[row * widthB + column] = sum;
}

__kernel void forwardPropagation(__global float* outputC, int widthA, int heightA, int widthB, int heightB, __global float* inputA, __global float* inputB, int bias, int activation) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  //int widthA, int heightA, int widthB, int heightB
  //printf("row: %d, col: %d\n", row, column);
  double sum = 0.0;
  for (int i = 0; i < widthA; i++) {
    sum += inputA[row * widthA + i] * inputB[i * widthB + column];
  }
  //activation 1 for tanh and 2 for exp
  if(activation == 1)
  	outputC[row * widthB + column] = tanh(sum + bias);
  else 
	outputC[row * widthB + column] = tanh(sum + bias);
}

