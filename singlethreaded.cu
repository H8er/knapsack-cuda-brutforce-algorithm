
		#include "cuda_runtime.h"
		#include "device_launch_parameters.h"
		#include "device_functions.h"
		#include <cstdlib>
		#include <stdlib.h>
		#include <sys/time.h>
		#include <iostream>

		#define arraySize 20
		using namespace std;

		__global__ void single_thread(int* bin_dev,int* weight_dev, int*values_dev,int W,int* str_num_dev,int num_of_blocks){
			int th_w_sum = 0;
			int th_v_sum = 0;
			int th_bin[arraySize];
			__shared__ int sh_w_d[arraySize];
			__shared__ int sh_v_d[arraySize];
			__shared__ int sh_maxs[1024];
			__shared__ int sh_we[1024];
			__shared__ int indices[1024];
			indices[threadIdx.x] = threadIdx.x;
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
					indices[threadIdx.x] = indices[threadIdx.x + s];
				}}
				__syncthreads();
			}
			// write result for this block to global mem
		bin_dev[blockIdx.x] = sh_maxs[0];
		bin_dev[blockIdx.x+num_of_blocks] = sh_we[0];
		str_num_dev[blockIdx.x] = indices[0]+blockIdx.x*blockDim.x;
		}

		__global__ void reduction_max(int* s,int* str_num_dev,int num_of_blocks) {
			extern __shared__ int sdata[];
			sdata[threadIdx.x+num_of_blocks] = str_num_dev[threadIdx.x];
			//red_ind[threadIdx.x] = str_num_dev[threadIdx.x];
			// each thread loads one element from global to shared mem
			//unsigned int tid = threadIdx.x;
			//unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
			sdata[threadIdx.x] = s[threadIdx.x];
			__syncthreads();
			// do reduction in shared mem
			for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
				if (threadIdx.x < s) {
					if (sdata[threadIdx.x] < sdata[threadIdx.x + s]){
						sdata[threadIdx.x] = sdata[threadIdx.x + s];
						//sdata[tid+64] = sdata[tid + s+64];
						sdata[threadIdx.x+num_of_blocks] = sdata[threadIdx.x+num_of_blocks + s];
				}}
				__syncthreads();
			}
			// write result for this block to global mem
			if (threadIdx.x == 0) {s[0] =sdata[0]; str_num_dev[0] = sdata[num_of_blocks];}//;s[1] = sdata[65];}
		}


		__global__ void which_string(int a,int*view_dev){
				view_dev[threadIdx.x] = (a>>threadIdx.x)%2;
		}
		int main()
		{
				int W = 350;

				struct timeval t0,t1;

				int totalSize = arraySize*pow(2,arraySize);
				int strSize_b = pow(2,arraySize);
				int num_of_blocks = strSize_b/1024;
				int *Sum=new int[num_of_blocks*2];// = { 0 };
				int *bin_dev;
				int *weight_dev;
				int weight[21] = { 5,10,17,19,20, 23,26,30,32,38, 40,44,47,50,55, 56,56,60,62, 66, 70	};
				int values[21] = { 10,13,16,22,30, 25,55,90,110,115, 130,120,150,170,194,199, 194, 199, 217, 230, 248	};
				int *values_dev;

				cudaMalloc((void**)&bin_dev, 2*num_of_blocks * sizeof(int));
				cudaMalloc((void**)&weight_dev, arraySize * sizeof(int));
				cudaMalloc((void**)&values_dev, arraySize * sizeof(int));

				int *str_num_dev;
				int*str_num = new int[num_of_blocks];
				cudaMalloc((void**)&str_num_dev,num_of_blocks*sizeof(int));

				cudaMemcpy(weight_dev, weight, arraySize * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(values_dev, values, arraySize * sizeof(int), cudaMemcpyHostToDevice);


		long sec,usec;

				int *w_sum;
				int *v_sum;

				cudaMalloc((void**)&w_sum, num_of_blocks * sizeof(int));
				cudaMalloc((void**)&v_sum, num_of_blocks * sizeof(int));


		gettimeofday(&t0, NULL);

		single_thread<<< num_of_blocks,1024>>>(bin_dev,weight_dev,values_dev,W,str_num_dev,num_of_blocks);
		reduction_max<<<1,num_of_blocks,num_of_blocks*sizeof(int)*2>>>(bin_dev,str_num_dev,num_of_blocks);

		gettimeofday(&t1, 0);
		cudaMemcpy(Sum, bin_dev, 2*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(str_num, str_num_dev, 1*sizeof(int), cudaMemcpyDeviceToHost);
		sec = (t1.tv_sec-t0.tv_sec);
		usec =  t1.tv_usec-t0.tv_usec;
		cout<<"GPU time = "<<sec<<" sec, "<<usec<<" microsec\n";

		cout<<"Acheived maximal sum = "<<Sum[0]<<"\n";
		cout<<"String number "<<str_num[0]<<"\n";
		cudaMemcpy(str_num, str_num_dev, 1*sizeof(int), cudaMemcpyDeviceToHost);
		int*view = new int[arraySize];
		int*view_dev;
		cudaMalloc((void**)&view_dev, arraySize*sizeof(int));
		which_string<<<1,arraySize>>>(str_num[0],view_dev);
		cudaMemcpy(view, view_dev, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0;i<arraySize;i++){
			cout<<view[i]<<" ";
		}cout<<"\n";
		//check
		int checksum = 0;
		for(int i = 0;i<arraySize;i++){
			checksum+=values[i]*view[i];
		}cout<<"Validation sum = "<<checksum<<"\n";
		checksum = 0;
		for(int i = 0;i<arraySize;i++){
			checksum+=weight[i]*view[i];
		}cout<<"Weight = "<<checksum<<"\n";
				cudaFree(bin_dev);
				cudaFree(weight_dev);
		    cudaFree(values_dev);
				cudaFree(view_dev);

		return 0;
		}
