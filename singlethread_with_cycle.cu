
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>

#define arraySize 30
#define threads_per_block 1024
#define max_blocks 32768
using namespace std;


__global__ void single_thread(int *sh_sum_dev, int *weight_dev, int *values_dev, int W, int *str_num_dev, int num_of_blocks,int rep)
{
  int th_w_sum = 0;
  int th_v_sum = 0;
  int th_bin[arraySize];
  __shared__ int sh_w_d[arraySize];
  __shared__ int sh_v_d[arraySize];
  __shared__ int sh_maxs[threads_per_block];
  __shared__ int indices[threads_per_block];
  indices[threadIdx.x] = threadIdx.x;

if(threadIdx.x<arraySize){
	sh_w_d[threadIdx.x] = weight_dev[threadIdx.x];
	sh_v_d[threadIdx.x] = values_dev[threadIdx.x];
}

__syncthreads();
int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
num_to_bin += max_blocks * rep;
#pragma unroll
  for (uint i = 0; i < arraySize; i++)
    {
      th_bin[i] = ((num_to_bin) >> i) % 2;
      th_w_sum += th_bin[i] * sh_w_d[i];
      th_v_sum += th_bin[i] * sh_v_d[i];
    }

sh_maxs[threadIdx.x] = (th_w_sum > W) ? 0:th_v_sum;


__syncthreads ();
  for (unsigned int offset = blockDim.x / 2; offset >= 1; offset >>= 1)
    {
      if (threadIdx.x < offset)
	{
	  if (sh_maxs[threadIdx.x] < sh_maxs[threadIdx.x + offset])
	    {
	      sh_maxs[threadIdx.x] = sh_maxs[threadIdx.x + offset];
	      indices[threadIdx.x] = indices[threadIdx.x + offset];
	    }
	}
      __syncthreads ();
    }
  // write result for this block to global mem
  sh_sum_dev[blockIdx.x+max_blocks*rep] = sh_maxs[0];
  //sh_sum_dev[blockIdx.x + num_of_blocks] = sh_we[0];
  str_num_dev[blockIdx.x+max_blocks*rep] = indices[0] + blockIdx.x * blockDim.x +max_blocks*rep;
}

__global__ void
reduction_max (int *s, int *str_num_dev)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int sdata[threads_per_block*2];
  sdata[threadIdx.x] = s[ID];
  sdata[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads ();
  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
	{
	  if (sdata[threadIdx.x] < sdata[threadIdx.x + s])
	    {
	      sdata[threadIdx.x] = sdata[threadIdx.x + s];
	      //sdata[tid+64] = sdata[tid + s+64];
	      sdata[threadIdx.x + threads_per_block] =
		sdata[threadIdx.x + threads_per_block + s];
	    }
	}
      __syncthreads ();
    }
  // write result for this block to global mem
  if (threadIdx.x == 0)
    {
			//if(sdata[0]>s[0]){//}&&(blockIdx.x>0)){
      s[blockIdx.x] = sdata[0];
      str_num_dev[blockIdx.x] = sdata[threads_per_block];
		}
    //}
}

__global__ void
which_string (int a, int *view_dev)
{
  view_dev[threadIdx.x] = (a >> threadIdx.x) % 2;
}

int main(){
  int W = 500;

  struct timeval t0, t1;

  int strSize_b = pow (2, arraySize);
  int num_of_blocks = strSize_b / threads_per_block;
  int *Sum = new int[1];	// = { 0 };
  int *sh_sum_dev;
  int *weight_dev;
  int weight[30] ={ 5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107 };
  int values[30] ={ 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313 };
  int *values_dev;
  long sec, usec;
  int *str_num_dev;
  int *str_num = new int[1];

  cout <<"N of items "<<arraySize<<"\n";
  cout<<"N of blocks "<<num_of_blocks<<"\n";
  cout<<"strSize_b = "<<strSize_b<<"\n";
  cout<<"num_of_blocks / threads_per_block = "<<num_of_blocks / threads_per_block<<"\n";
  cout<<"IDEA  "<<strSize_b/num_of_blocks<<"\n";
  cout<<"red param "<<1 + (num_of_blocks-1) / threads_per_block<<"  ,  "<<strSize_b/num_of_blocks<<"\n";

  int N_of_rep;
  N_of_rep = num_of_blocks/max_blocks;
  cout<<"N of repeats = "<<N_of_rep<<"\n";
  cout<<"sing param = "<<num_of_blocks/N_of_rep<<" _ "<< threads_per_block<<"\n";
  gettimeofday (&t0, NULL);
  cudaMalloc ((void **) &sh_sum_dev,  num_of_blocks * sizeof (int));
  cudaMalloc ((void **) &weight_dev, arraySize * sizeof (int));
  cudaMalloc ((void **) &values_dev, arraySize * sizeof (int));
  cudaMalloc ((void **) &str_num_dev, num_of_blocks * sizeof (int));

  cudaMemcpy (weight_dev, weight, arraySize * sizeof (int),
	      cudaMemcpyHostToDevice);
  cudaMemcpy (values_dev, values, arraySize * sizeof (int),
	      cudaMemcpyHostToDevice);


        for(int i = 0;i<N_of_rep;i++){
  single_thread <<< num_of_blocks/N_of_rep, threads_per_block >>> (sh_sum_dev, weight_dev, values_dev,
					     W, str_num_dev, num_of_blocks,i);
             }
//cout all the parameters


  reduction_max <<<1 + (num_of_blocks-1) / threads_per_block, strSize_b/num_of_blocks>>> (sh_sum_dev, str_num_dev);
  reduction_max <<<1, 1 + (num_of_blocks-1) / threads_per_block>>> (sh_sum_dev, str_num_dev);

  cudaMemcpy (Sum, sh_sum_dev, sizeof (int), cudaMemcpyDeviceToHost);
  cudaMemcpy (str_num, str_num_dev, sizeof (int), cudaMemcpyDeviceToHost);
	gettimeofday (&t1, 0);
	sec = (t1.tv_sec - t0.tv_sec);
  usec = t1.tv_usec - t0.tv_usec;
  cout << "GPU time = " << sec << " sec, " << usec << " microsec\n";

  cout << "Acheived maximal sum = " << Sum[0] << "\n";
  cout << "String number " << str_num[0] << "\n";

  int *view = new int[arraySize];
  int *view_dev;
  cudaMalloc ((void **) &view_dev, arraySize * sizeof (int));
  which_string <<< 1, arraySize >>> (str_num[0], view_dev);
  cudaMemcpy (view, view_dev, arraySize * sizeof (int),
	      cudaMemcpyDeviceToHost);
  for (int i = 0; i < arraySize; i++)
    {
      cout << view[i] << " ";
    } cout << "\n";
  //check
  int checksum = 0;
  for (int i = 0; i < arraySize; i++)
    {
      checksum += values[i] * view[i];
    } cout << "Validation sum = " << checksum << "\n";
  checksum = 0;
  for (int i = 0; i < arraySize; i++)
    {
      checksum += weight[i] * view[i];
    } cout << "Weight = " << checksum << "\n";
  cudaFree (sh_sum_dev);
  cudaFree (weight_dev);
  cudaFree (values_dev);
  cudaFree (view_dev);

  return 0;
}
