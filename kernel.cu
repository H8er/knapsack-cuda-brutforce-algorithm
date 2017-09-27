
//Compile in MSVS15 as it is
//Compile in Linux-systems using "nvcc -ccbin clang kernel.cu"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <iostream>


using namespace std;

//bin_dev, wight_dev, s_dev, values_dev);
__global__ void smth(int *bin_dev, int* wight_dev, int *s_dev, int*values_dev)
{					//bin_dev,wight_dev,s_dev,values_dev
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	s_dev[i] = bin_dev[i] * values_dev[threadIdx.x];//s_dev -> prices*bin
	bin_dev[i] = bin_dev[i] * wight_dev[threadIdx.x];//bin_dev -> weights*bin
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


__global__ void T_binary3(int*bin_dev, int *_del) {
	int bli = blockIdx.x * blockDim.x;
	int idx = threadIdx.x;
	bin_dev[bli + idx] = blockIdx.x / _del[idx] % 2;
}
__global__ void summing(int *s_dev, int*w, int*s, int*bin_dev, int W) {
	int bli = blockIdx.x * blockDim.x;
	for (int i = 0; i < 15; i++) {
		w[blockIdx.x] += bin_dev[bli + i];
		s[blockIdx.x] += s_dev[bli + i];
	}
	if (w[blockIdx.x] > W) { s[blockIdx.x] = 0; w[blockIdx.x] = 0; }
	//15x32x1024

}

__global__ void plusing2_w(int* in_dev, int* sums) {
	__shared__ int sdata[15];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = in_dev[i];
	__syncthreads();
	//unroled summing is faster than reduction according to my tests

sdata[0] += sdata[1];
sdata[0] += sdata[2];
sdata[0] += sdata[3];
sdata[0] += sdata[4];
sdata[0] += sdata[5];
sdata[0] += sdata[6];
sdata[0] += sdata[7];
sdata[0] += sdata[8];
sdata[0] += sdata[9];
sdata[0] += sdata[10];
sdata[0] += sdata[11];
sdata[0] += sdata[12];
sdata[0] += sdata[13];
sdata[0] += sdata[14];

		__syncthreads();
	
	// write result for this block to global mem
	if (tid == 0) sums[blockIdx.x] = sdata[0];
}

__global__ void zeroing(int *w, int *s, int W) {
	int bli = blockIdx.x * blockDim.x;
	int idx = threadIdx.x;
	if (w[bli+idx] > W) { s[bli + idx] = 0; w[bli + idx] = 0; }
}
__global__ void max1(int* s) {
	__shared__ int sdata[1024];
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

int main()
{
	int qwerty = 0;
	while (qwerty < 1)
	{

		int W = 300;
		int arraySize = 15;
		int arraySize_b = 32768 * 15;
		int Sum[32768 * 15] = { 0 };
		int *s;
		//int bin[32768 * 15];
		int *bin_dev;
		int *wight_dev;
		//int prices[32768 * 15] = { 0 };
		int wight[15] = { 5,10,17,19,20, 23,26,30,32,38,   40,44,47,50,55 };// 55, 56, 60, 62, 66, 70	};
		int values[15] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194 };// , 194, 199, 217, 230, 248	};
		int *w;
		int *values_dev;


		int del[15], *_del;
		cudaMalloc((void**)&_del, 15 * sizeof(int));
		for (int i = 0; i < 15; i++) {
			del[i] = pow(2, i);
		}

		float gpu_elapsed_time,sumoftime=0;
/*
		cudaEvent_t gpu_start, gpu_stop;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);
*/
		cudaMemcpy(_del, del, 15 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&bin_dev, arraySize_b * sizeof(int));

		int*s_dev;
		cudaMalloc((void**)&s_dev, arraySize_b * sizeof(int));
		cudaMalloc((void**)&wight_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&s, arraySize_b * sizeof(int));
		cudaMalloc((void**)&values_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&w, 32768 * sizeof(int));
		cudaMemcpy(wight_dev, wight, arraySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values_dev, values, arraySize * sizeof(int), cudaMemcpyHostToDevice);

		cudaEvent_t gpu_start, gpu_stop;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		T_binary3 << <32768, 15 >> > (bin_dev, _del);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "T_binary3: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		smth << <32768,15 >> > (bin_dev, wight_dev, s_dev, values_dev);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "smth: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		plusing2_w << <32768, 15 >> > (bin_dev, w);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "plusing_weight: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		plusing2_w << <32768, 15 >> > (s_dev, s);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "plusing_prices: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		zeroing << <32, 1024 >> > (w, s, W);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "zeroing: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		max1 << <32, 1024 >> > (s);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "max1: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;



		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		for (int i = 32; i >= 1; i /= 2) {
			kermax2 << <1, i >> > (s,i);
			//cudaThreadSynchronize();

		}

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "loops of kermax2: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		cudaMemcpy(Sum, s, sizeof(int), cudaMemcpyDeviceToHost);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "cpy of maximal sum: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;
		cout <<"\n"<<"max = " << Sum[0] << "\n";

		
		cout << "Total time: " << sumoftime << " milli-seconds\n";

		cudaFree(bin_dev);
		cudaFree(wight_dev);
		cudaFree(s);
		cudaFree(w);
		cudaFree(s_dev);
		cout << "\n";


		//system("pause");

		//CPU version 1
		float fTimeStart = clock() / (float)(CLOCKS_PER_SEC);
		int bin[32768][15] = { 0 };
		int k = 0;
		for (int i = 0; i < 32768; i++) {
			k = i;
			for (int j = 14; j >= 0; j--) {
				bin[i][j] = k % 2;
				k /= 2;
			}
		}
		//int wight[15] = { 5,10,17,19,20, 23,26,30,32,38,   40,44,47,50,55 };// 55, 56, 60, 62, 66, 70	};
		//int values[15] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194 };// , 194, 199, 217, 230, 248	};
		int prices[32768][15] = { 0 };
		int weights[32768][15] = { 0 };
		int Sweig[32768] = { 0 };
		int Sval[32768] = { 0 };
		for (int i = 0; i < 32768; i++) {
			for (int j = 0; j < 15; j++) {
				weights[i][j] = wight[j] * bin[i][j];
				prices[i][j] = values[j] * bin[i][j];
			}
		}
		for (int i = 0; i < 32768; i++) {
			for (int j = 0; j < 15; j++) {
				Sweig[i] += weights[i][j];
				Sval[i] += prices[i][j];
			}
		}
		int max = 0; k = 0;
		for (int i = 0; i < 32768; i++) {
			if ((Sweig[i] <= W) && (Sval[i] > max)) {
				k = i; max = Sval[i];
			}
		}
		float fTimeStop = clock() / (float)CLOCKS_PER_SEC;
		cout << "\nmax = " << max << "\n" << "String No " << k << "\n";
		//for (int i = 0; i < 10; i++) { cout << bin[k][i] << " "; }cout << "\n";
		//for (int i = 0; i < 10; i++) { if (weights[k][i] != 0)cout << weights[k][i] << " + "; }cout << "\n";
		cout << "CPU time is " << (fTimeStop - fTimeStart) * 1000 << " milli-seconds\n";

		qwerty++;
	}
	

	return 0;
}
