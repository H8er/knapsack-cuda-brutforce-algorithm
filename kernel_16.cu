
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <iostream>


using namespace std;

__global__ void T_binary3(int*bin_dev, int *_del) {
	int bli = blockIdx.x * blockDim.x;
	int idx = threadIdx.x;
	bin_dev[bli + idx] = blockIdx.x / _del[idx] % 2;
	bin_dev[bli+idx+32768*16] = bin_dev[bli+idx];
	bin_dev[bli+15+32768*16] = 1;
}


__global__ void smth(int *bin_dev, int* wight_dev, int *s_dev, int*values_dev)
{					//bin_dev,wight_dev,s_dev,values_dev
	int bli = blockIdx.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_dev[i] = bin_dev[i] * values_dev[threadIdx.x];//s_dev -> prices*bin
	s_dev[i+32768*16] = s_dev[i];
	s_dev[bli+15+32768*16] = values_dev[15];
	bin_dev[i] = bin_dev[i] * wight_dev[threadIdx.x];//bin_dev -> weights*bin
	bin_dev[i+32768*16] = bin_dev[i];
	bin_dev[bli+15+32768*16] = wight_dev[15];
}

__global__ void plusing2_w(int* in_dev, int* sums) {
	__shared__ int sdata[16];
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
//sdata[0] += sdata[15];

		__syncthreads();

	// write result for this block to global mem
	if (tid == 0) sums[blockIdx.x] = sdata[0];
}
__global__ void plusing2_w2(int* add, int* sums) {

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sums[i+32768]=sums[i]+add[15];
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
	int qwerty = 0;
	while (qwerty < 1)
	{

		int W = 300;
		int arraySize = 16;
		int arraySize_b = 32768*2 * 16;
		int Sum[65536*16] = { 0 };
		int *s;
		int *bin_dev;
		int *wight_dev;
		int wight[16] = { 5,10,17,19,20, 23,26,30,32,38,   40,44,47,50,55 ,56};// 55, 56, 60, 62, 66, 70	};
		int values[16] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194,199 };// , 194, 199, 217, 230, 248	};
		int *w;
		int *values_dev;


		int del[16], *_del;
		cudaMalloc((void**)&_del, 16 * sizeof(int));
		for (int i = 0; i < 16; i++) {
			del[i] = pow(2, i);cout<<del[i]<<" ";
		}

		float gpu_elapsed_time,sumoftime=0;

		cudaEvent_t gpu_start, gpu_stop;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		cudaMemcpy(_del, del, 16 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&bin_dev, arraySize_b * sizeof(int));

		int*s_dev;
		cudaMalloc((void**)&s_dev, arraySize_b * sizeof(int));
		cudaMalloc((void**)&wight_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&s, arraySize_b * sizeof(int));
		cudaMalloc((void**)&values_dev, arraySize * sizeof(int));
		cudaMalloc((void**)&w, 32768 *2* sizeof(int));
		cudaMemcpy(wight_dev, wight, arraySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values_dev, values, arraySize * sizeof(int), cudaMemcpyHostToDevice);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
sumoftime+=gpu_elapsed_time;

cudaEventCreate(&gpu_start);
cudaEventCreate(&gpu_stop);
cudaEventRecord(gpu_start, 0);

		cout << "Copying all needed data to GPU: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		T_binary3 << <32768, 16 >> > (bin_dev, _del);
/*
cudaMemcpy(Sum, bin_dev, 32768*2*16*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<5;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j];}cout<<"\n";
}cout<<"---------------\n";
for(int i=32763;i<32768;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j];}cout<<"\n";
}cout<<"---------------\n";
for(int i=32768;i<32773;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j];}cout<<"\n";
}
cout<<"---------------\n";
for(int i=65530;i<65536;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j];}cout<<"\n";
}
*/

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

		smth << <32768,16 >> > (bin_dev, wight_dev, s_dev, values_dev);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "smth: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;
/*
cout<<"-|-|-|-|-|-|-|-|-|-\n";
for(int j=0;j<16;j++){cout<<values[j]<<" ";}cout<<"\n";
cout<<"-|-|-|-|-|-|-|-|-|-\n";
cudaMemcpy(Sum, s_dev, 32768*2*16*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<5;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j]<<" ";}cout<<"\n";
}cout<<"---------------\n";
for(int i=32763;i<32768;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j]<<" ";}cout<<"\n";
}cout<<"---------------\n";
for(int i=32768;i<32773;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j]<<" ";}cout<<"\n";
}
cout<<"---------------\n";
for(int i=65530;i<65536;i++){
	for(int j=0;j<16;j++){cout<<Sum[i*16+j]<<" ";}cout<<"\n";
}

*/

		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		plusing2_w << <32768, 16 >> > (bin_dev, w);
		plusing2_w2 << <32,1024 >> > (wight_dev,w);
		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "plusing_weight: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;

/*
cudaMemcpy(Sum, w, 65536*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<65536;i++){cout<<Sum[i]<<" ";}
*/

		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		plusing2_w << <32768, 16 >> > (s_dev, s);
	  plusing2_w2 << <32,1024 >> > (values_dev,s);
		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "plusing_prices: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;

/*
cudaMemcpy(Sum, s, 65536*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<65536;i++){cout<<Sum[i]<<" ";}
*/
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		zeroing << <64, 1024 >> > (w, s, W);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "zeroing: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;

/*
cudaMemcpy(Sum, s, 65536*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<65536;i++){cout<<Sum[i]<<" ";}
*/

		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		max1 << <64, 1024 >> > (s);

		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		cout << "max1: " << gpu_elapsed_time << " milli-seconds\n";
sumoftime+=gpu_elapsed_time;
gpu_elapsed_time = 0;

/*
cudaMemcpy(Sum, s, 64*sizeof(int), cudaMemcpyDeviceToHost);
for(int i=0;i<64;i++){cout<<Sum[i]<<" ";}
*/


		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);
kermax2 << <2, 32 >> > (s,32);
		for (int i = 32; i >= 1; i /= 2) {
			kermax2 << <1, i >> > (s,i);
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
		//int bin[32768*16] = { 0 };
		int k = 0;
		for (int i = 0; i < 32768; i++) {
			k = i;
			for (int j = 15; j >= 0; j--) {
				Sum[i*16+j] = k % 2;
				k /= 2;
			}
		}

		//int wight[15] = { 5,10,17,19,20, 23,26,30,32,38,   40,44,47,50,55 };// 55, 56, 60, 62, 66, 70	};
		//int values[15] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194 };// , 194, 199, 217, 230, 248	};
		//int prices[32768][16] = { 0 };
		//int weights[32768][16] = { 0 };
		int Sweig[65536] = { 0 };
		int Sval[65536] = { 0 };


		for (int i = 0; i < 32768; i++) {
			for (int j = 0; j < 16; j++) {
				Sum[i*16+j] = wight[j] * Sum[i*16+j];
				//prices[i][j] = values[j] * Sum[i*16+j];
			}
		}

		for (int i = 0; i < 32768; i++) {
			for (int j = 0; j < 16; j++) {
				Sweig[i] +=Sum[i*16+j];;
				//Sval[i] += prices[i][j];
			}
		}
		for (int i = 32768; i < 65536; i++) {
			for (int j = 0; j < 16; j++) {
				Sweig[i] =Sweig[i-32768]+ wight[15];
				//Sval[i] =Sval[i]+values[15];
			}
		}
		for (int i = 0; i < 32768; i++) {
			for (int j = 0; j < 16; j++) {
//			Sum[i*16+j] = wight[j] * Sum[i*16+j];
				Sum[i*16+j] = values[j] * Sum[i*16+j];
			}
		}
		for (int i = 32768; i < 65536; i++) {
			for (int j = 0; j < 16; j++) {
				//Sweig[i] =Sweig[i-32768]+ wight[15];
				Sval[i] =Sval[i-32768]+values[15];
			}
		}

		int max = 0; k = 0;
		for (int i = 0; i < 65536; i++) {
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
