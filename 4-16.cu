
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>


using namespace std;

__global__ void T_binary(int*bin_dev, int *_del) {
	int bli = blockIdx.x * blockDim.x;
	int idx = threadIdx.x;
	bin_dev[bli + idx] = blockIdx.x / _del[idx] % 2;
}

__global__ void bin_multiplication(int *bin_dev, int* wight_dev, int *s_dev, int*values_dev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_dev[i] = bin_dev[i] * values_dev[threadIdx.x];//s_dev -> prices*bin
	bin_dev[i] = bin_dev[i] * wight_dev[threadIdx.x];//bin_dev -> weights*bin
}

__global__ void summing(int* in_dev, int* sums) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = in_dev[i];
	//__syncthreads();
	//unroled summing is faster than reduction according to my tests
#pragma unroll
for(ushort  i = 1;i < 16;i++){
	sdata[0]+=sdata[i];
}
		__syncthreads();
	// write result for this block to global mem
	sums[blockIdx.x] = sdata[0];
}

__global__ void additional_summing(int *whatToAdd,int *whereToAdd){
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	whereToAdd[i+32768] = whereToAdd[i]+whatToAdd[15];
}


__global__ void zeroing(int *w, int *s, int W) {
	int bli = blockIdx.x * blockDim.x;
	int idx = threadIdx.x;
	if (w[bli+idx] > W) { s[bli + idx] = 0; w[bli + idx] = 0; }
}
__global__ void reduction_max(int* s) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = s[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			if (sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) s[blockIdx.x] = sdata[0];
}

__global__ void kermax2(int *s, int N) {
	//__shared__ int max[32];
	unsigned int tid = threadIdx.x;
	int off = N / 2;
	if (tid < off) {
		if (s[tid] < s[tid + off]) {
			s[tid] = s[tid + off];
		}
	}
}

int main()
{
		int W = 350;
		int arraySize;
		cout<<"Enter size of array (6-15): ";
		cin>>arraySize;		

		struct timeval t0,t1;
			gettimeofday(&t0, NULL);

		int totalSize = arraySize*pow(2,arraySize);
		int strSize_b = pow(2,arraySize);
		int flag=0;
		if (arraySize>15){
			strSize_b/=(pow(2,(arraySize-15)));
			flag=1;
		}
		int *Sum=new int[totalSize];// = { 0 };
		int *s;
		int *bin_dev;
		int *wight_dev;
		int wight[16] = { 5,10,17,19,20, 23,26,30,32,38, 40,44,47,50,55,56 };// 55, 56, 60, 62, 66, 70	};
		int values[16] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194,199 };// , 194, 199, 217, 230, 248	};
		int *w;
		int *values_dev;

		int *del = new int[arraySize], *dev_del;
		cudaMalloc((void**)&dev_del, arraySize * sizeof(int));
		for (int i = 0; i < arraySize; i++) {
			del[i] = pow(2, i);
		}

		cudaMemcpy(dev_del, del, arraySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&bin_dev, totalSize * sizeof(int));

		int*s_dev;
		cudaMalloc((void**)&s_dev, totalSize * sizeof(int));
		cudaMalloc((void**)&wight_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&s, totalSize * sizeof(int));
		cudaMalloc((void**)&values_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&w, totalSize * sizeof(int));
		cudaMemcpy(wight_dev, wight, arraySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values_dev, values, arraySize * sizeof(int), cudaMemcpyHostToDevice);

//creating of binary table
		T_binary << <strSize_b, arraySize >> > (bin_dev, dev_del);

//multiplication of weight and value parameters of each item on binary table strings
		bin_multiplication << <strSize_b, arraySize >> > (bin_dev, wight_dev, s_dev, values_dev);

		summing << <strSize_b, arraySize,arraySize*sizeof(int) >> > (bin_dev, w);
		summing << <strSize_b, arraySize,arraySize*sizeof(int) >> > (s_dev, s);

		int a=strSize_b/1024;
		int b = 1024;
		if (a==0){
			a=1;
			b = pow(2,arraySize);}

		//additional actions if flag==1
		if(flag==1){
additional_summing<<<a, b>>>(wight_dev,w);
additional_summing<<<a, b>>>(values_dev,s);
		}

//zeroing of unsuitable item's combinations



		zeroing << <a, b >> > (w, s, W);

//finding maximal value for each block
		reduction_max << <a,b,b*sizeof(int) >> > (s);
		cudaMemcpy(Sum, s, a*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0;i<a;i++){cout<<Sum[i]<<" ";}
//second step of finding maximal value
		for (int i = 32; i >= 1; i /= 2) {
			kermax2 << <1, i >> > (s,i);
		}

		cudaMemcpy(Sum, s, sizeof(int), cudaMemcpyDeviceToHost);

		cout <<"\n"<<"GPU max = " << Sum[0];

		cudaFree(bin_dev);
		cudaFree(wight_dev);
		cudaFree(s);
		cudaFree(w);
		cudaFree(s_dev);

		//CPU version
		float fTimeStart = clock() / (float)(CLOCKS_PER_SEC);
		int **bin = new int*[strSize_b];
			for(int i=0;i<strSize_b;i++){
				bin[i] = new int[arraySize];
			}
		int k = 0;
		for (int i = 0; i < strSize_b; i++) {
			k = i;
			for (int j = 0; j <arraySize; j++) {
				bin[i][j] = k % 2;
				k /= 2;
			}
		}

		int **prices  = new int*[strSize_b];
		int **weights = new int*[strSize_b];
			for(int i=0;i<strSize_b;i++){
				prices[i]  = new int[arraySize];
				weights[i] = new int[arraySize];
			}

		int *Sweig = new int[strSize_b];
		int *Sval = new int[strSize_b];
		for (int i = 0; i < strSize_b; i++) {
			for (int j = 0; j < arraySize; j++) {
				weights[i][j] = wight[j] * bin[i][j];
				prices[i][j] = values[j] * bin[i][j];
			}
		}
		for (int i = 0; i < strSize_b; i++) {
			for (int j = 0; j < arraySize; j++) {
				Sweig[i] += weights[i][j];
				Sval[i] += prices[i][j];
			}
		}
		int max = 0; k = 0;
		for (int i = 0; i < strSize_b; i++) {
			if ((Sweig[i] <= W) && (Sval[i] > max)) {
				k = i; max = Sval[i];
			}
		}
		float fTimeStop = clock() / (float)CLOCKS_PER_SEC;
		cout << "   CPU max = " << max << "\n";
		//cout << "CPU time is " << (fTimeStop - fTimeStart) * 1000 << " milli-seconds\n";
//memory freeing
		for(int i=0;i<strSize_b;i++){
			delete [] bin[strSize_b];
			delete [] prices[strSize_b];
			delete [] weights[strSize_b];
		}
		delete [] Sweig;
		delete [] Sval;

gettimeofday(&t1, 0);
long sec = (t1.tv_sec-t0.tv_sec);
long usec =  t1.tv_usec-t0.tv_usec;
cout<<sec<<","<<usec;
return 0;
}
