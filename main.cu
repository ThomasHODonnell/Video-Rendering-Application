#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void testKernel() {
	printf("%s", "Hello from the GPU!\n");
}

int main() {
	testKernel << <1, 1 >> > ();
	cudaDeviceSynchronize();
	printf("%s", "Hello from the CPU!\n");
	return 0;
}