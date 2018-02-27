
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>

#define arraySize 16
using namespace std;

__global__ void single_thread(int* bin_dev,int* weight_dev, int*values_dev,int W,int* NO){
	int th_w_sum = 0;
	int th_v_sum = 0;
	int th_bin[arraySize];
	__shared__ int sh_w_d[arraySize];
	__shared__ int sh_v_d[arraySize];
	__shared__ int sh_maxs[1024];
	__shared__ int sh_we[1024];

	#pragma unroll
	for(uint i = 0;i<arraySize;i++){
		th_bin[i] = ((blockIdx.x*blockDim.x + threadIdx.x)>>i)%2;
		sh_w_d[i] = weight_dev[i];
		sh_v_d[i] = values_dev[i];
		th_w_sum+=th_bin[i]*sh_w_d[i];
		th_v_sum+=th_bin[i]*sh_v_d[i];
	}

	sh_maxs[threadIdx.x] = th_v_sum;
	sh_we[threadIdx.x] = th_w_sum;
	__syncthreads();
	if(sh_we[threadIdx.x]>W){sh_maxs[threadIdx.x]=0;sh_we[threadIdx.x]=0;
	}

	for (unsigned int s = blockDim.x / 2; s>=1; s >>= 1) {
		if(threadIdx.x<s){
			if(sh_maxs[threadIdx.x]<sh_maxs[threadIdx.x + s]){
			sh_maxs[threadIdx.x]=sh_maxs[threadIdx.x + s];
			sh_we[threadIdx.x] = sh_we[threadIdx.x + s];
		}}
		__syncthreads();
	}
	// write result for this block to global mem
bin_dev[blockIdx.x] = sh_maxs[0];
bin_dev[blockIdx.x+64] = sh_we[0];
}

__global__ void reduction_max(int* s) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = s[tid];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			if (sdata[tid] < sdata[tid + s]){
				sdata[tid] = sdata[tid + s];
				//sdata[tid+64] = sdata[tid + s+64];
		}}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) {s[0] =sdata[0];}//;s[1] = sdata[65];}
}

int main()
{
		int W = 350;

		struct timeval t0,t1;

		int totalSize = arraySize*pow(2,arraySize);
		int strSize_b = pow(2,arraySize);

		int *Sum=new int[strSize_b/1024*2];// = { 0 };
		int *bin_dev;
		int *weight_dev;
		int weight[16] = { 5,10,17,19,20, 23,26,30,32,38, 40,44,47,50,55,56 };// 55, 56, 60, 62, 66, 70	};
		int values[16] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194,199 };// , 194, 199, 217, 230, 248	};
		int *values_dev;

		cudaMalloc((void**)&bin_dev, 2*strSize_b/1024 * sizeof(int));
		cudaMalloc((void**)&weight_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&values_dev, arraySize * sizeof(int));

		cudaMemcpy(weight_dev, weight, arraySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values_dev, values, arraySize * sizeof(int), cudaMemcpyHostToDevice);


long sec,usec;

		int *w_sum;
		int *v_sum;
		int *NO;
		cudaMalloc((void**)&w_sum, strSize_b/1024 * sizeof(int));
		cudaMalloc((void**)&v_sum, strSize_b/1024 * sizeof(int));
		cudaMalloc((void**)&NO, sizeof(int));

gettimeofday(&t0, NULL);
single_thread<<< strSize_b/1024,1024>>>(bin_dev,weight_dev,values_dev,W,NO);
reduction_max<<<1,strSize_b/1024,strSize_b/1024*sizeof(int)>>>(bin_dev);
gettimeofday(&t1, 0);
cudaMemcpy(Sum, bin_dev, 2*sizeof(int), cudaMemcpyDeviceToHost);
sec = (t1.tv_sec-t0.tv_sec);
usec =  t1.tv_usec-t0.tv_usec;
cout<<"GPU time = "<<sec<<" sec, "<<usec<<" microsec\n";

cout<<"Acheived maximal sum = "<<Sum[0]<<"\n";


		cudaFree(bin_dev);
		cudaFree(weight_dev);
    cudaFree(values_dev);
		cudaFree(NO);

return 0;
}
