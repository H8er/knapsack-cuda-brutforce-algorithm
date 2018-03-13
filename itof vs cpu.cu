
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>

#define arraySize 31 //35 max
#define threads_per_block 1024
#define max_blocks 32768
using namespace std;

__constant__ float coefs[arraySize*2];
__global__ void single_thread(float *sh_sum_dev,float W, float *str_num_dev, float num_of_blocks,int rep)
{
  register float th_w_sum = 0;
  register float th_v_sum = 0;
  register float th_bin[arraySize];
  register int n_of_it=rep;

  __shared__ float sh_maxs[threads_per_block];
  __shared__ float indices[threads_per_block];
  indices[threadIdx.x] = threadIdx.x;
  __syncthreads();

long int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
num_to_bin += max_blocks * n_of_it;
#pragma unroll
  for (uint i = 0; i < arraySize; i++)
    {
      th_bin[i] = ((num_to_bin) >> i) % 2;

      th_w_sum += th_bin[i] * coefs[i];
      th_v_sum += th_bin[i] * coefs[i+arraySize];
    }

sh_maxs[threadIdx.x] = (th_w_sum > W) ? 0:th_v_sum;

__syncthreads ();
  for (uint offset = blockDim.x / 2; offset >= 1; offset >>= 1)
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
  if(threadIdx.x == 0){
  sh_sum_dev[blockIdx.x+max_blocks*rep] = sh_maxs[0];
  str_num_dev[blockIdx.x+max_blocks*rep] = indices[0] + blockIdx.x * blockDim.x +max_blocks*rep;
}}

__global__ void
reduction_max (float *s, float *str_num_dev)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[threads_per_block*2];
  sdata[threadIdx.x] = s[ID];
  sdata[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads ();
  // do reduction in shared mem
  for (uint s = blockDim.x / 2; s > 0; s >>= 1)
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
  float W = 500;

  struct timeval t0, t1;
  cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
float milliseconds1 = 0;
float milliseconds2 = 0;
  long int strSize_b = pow (2, arraySize);
  int num_of_blocks = strSize_b / threads_per_block;
  float *Sum = new float[1];	// = { 0 };
  float *sh_sum_dev;
  float weight[31] ={ 5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107,115 };
  float values[31] ={ 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313,321 };
  float dev_coefs[62] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107,115, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313,321 };
  //float dev_coefs[60] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313 };

  //float *values_dev;
  long sec, usec;
  float *str_num_dev;
  float *str_num = new float[1];
  float N_of_rep;
  N_of_rep = num_of_blocks/max_blocks;

  cout <<"N of items "<<arraySize<<"\n";
  cout<<"N of blocks "<<num_of_blocks<<"\n";
  cout<<"strSize_b = "<<strSize_b<<"\n";
  cout<<"num_of_blocks / threads_per_block = "<<num_of_blocks / threads_per_block<<"\n";
  cout<<"N of repeats = "<<N_of_rep<<"\n";
  cout<<"sing param = "<<num_of_blocks/N_of_rep<<" _ "<< threads_per_block<<"\n";
  cout<<"red param "<<num_of_blocks / threads_per_block<<"  ,  "<<strSize_b/num_of_blocks<<"\n";


  gettimeofday (&t0, NULL);
  cudaMalloc ((void **) &sh_sum_dev,  num_of_blocks * sizeof (float));


  cudaMalloc ((void **) &str_num_dev, num_of_blocks * sizeof (float));

  cudaMemcpyToSymbol (coefs, dev_coefs, 2*arraySize * sizeof (float));

cudaEventRecord(start);
int sing_blocks = num_of_blocks/N_of_rep;
        for(int i = 0;i<N_of_rep;i++){
  single_thread <<< sing_blocks, threads_per_block >>> (sh_sum_dev, W, str_num_dev, num_of_blocks,i);
             }
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&milliseconds1, start, stop);
cudaEventRecord(start);
int k = num_of_blocks/threads_per_block;
while(k>1){

               reduction_max <<<k, threads_per_block>>> (sh_sum_dev, str_num_dev);
               if(k>=threads_per_block){k/=threads_per_block;}
               else break;
             }

reduction_max <<<1,k>>> (sh_sum_dev, str_num_dev);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&milliseconds2, start, stop);
  cudaMemcpy (Sum, sh_sum_dev, sizeof (float), cudaMemcpyDeviceToHost);
  cudaMemcpy (str_num, str_num_dev, sizeof (float), cudaMemcpyDeviceToHost);
	gettimeofday (&t1, 0);
	sec = (t1.tv_sec - t0.tv_sec);
  usec = t1.tv_usec - t0.tv_usec;
  cout << "GPU time = " << sec << " sec, " << usec << " microsec\n";
  cout<<"single_thread time = "<<milliseconds1/1000.0<<"sec\n";
  cout<<"reduction time = "<<milliseconds2/1000.0<<"sec\n";

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
  float checksum = 0;
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
  cudaFree (str_num_dev);
  cudaFree (coefs);
  cudaFree (view_dev);


cout<<"CPU version:\n";

float *cpu_bin = new float[arraySize];
int max = 0;
int tmp = 0;
int cpu_str = 0;
int cap;
gettimeofday (&t0, NULL);
for(long int i = 0;i<num_of_blocks;i++){
  for(int j = 0; j<threads_per_block;j++){
    int tobin = i*threads_per_block+j;
    for(int k = 0; k<arraySize;k++){
      cpu_bin[k] = tobin%2;
      tobin>>=1;
      tmp += cpu_bin[k]*values[k];
      cap += cpu_bin[k]*weight[k];
    }
    if((cap<=W)&&(tmp>max)){max = tmp;cpu_str = i*threads_per_block+j;}
    tmp = 0; cap = 0;
}
}

gettimeofday (&t1, 0);
sec = (t1.tv_sec - t0.tv_sec);
usec = t1.tv_usec - t0.tv_usec;
cout<<"Max = "<<max<<"\n"<<"STR = "<<cpu_str<<"\n";
cout << "CPU time = " << sec << " sec, " << usec << " microsec\n";

  return 0;
}
