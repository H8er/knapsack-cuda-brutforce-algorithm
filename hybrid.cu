
#include "cuda_runtime.h"
#include <iostream>
#include <chrono>

#define arraySize 29 //35 max
#define def_div 10
#define W 31
#define threads_per_block 32
#define max_blocks 32

using namespace std;

__constant__ float coefs[arraySize*2];
__global__ void single_thread(float *sh_sum_dev, long int *str_num_dev, float num_of_blocks, int* bdevX,int* global_mem_bin)
{
  float th_w_sum = 0;
   float th_v_sum = 0;
   int th_bin[arraySize];
   int best_bin[arraySize];
  __shared__ float sh_maxs[threads_per_block];
  __shared__ long int indices[threads_per_block];
  int reached = 0;
  indices[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();

long signed int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
//num_to_bin += max_blocks * n_of_it;
#pragma unroll
  for (uint i = 0; i < def_div; i++)
    {
      th_bin[i] = ((num_to_bin) >> i) % 2;
      th_w_sum += th_bin[i] * coefs[i];
      th_v_sum += th_bin[i] * coefs[i+arraySize];
      best_bin[i] = th_bin[i];
    }
#pragma unroll
    for (uint i = def_div; i < arraySize; i++)
      {
        th_bin[i] = -1;
      }
__syncthreads ();
register int Capacity = W - th_w_sum;
sh_maxs[threadIdx.x] = (th_w_sum > W) ? 0:th_v_sum;
__syncthreads ();


//H_S
int h = def_div;
long int ns = 0;
bool forward;

while(h-def_div!=-1){
  ns++;
  forward = true;
  if(th_bin[h]==-1){
     th_bin[h]=1;
  }else{
  if(th_bin[h]==1){
     th_bin[h]=0;
  }else{
  if(th_bin[h]==0){
     th_bin[h]=-1;
    h--;
    forward=false;
  }
}
}
  if(h==arraySize-1){
    int cw = 0;
    int cp = 0;
    #pragma unroll
    for(int i = def_div;i<arraySize;i++){
      cp += coefs[i+arraySize] * th_bin[i];
      cw += coefs[i] * th_bin[i];
    }
    if((cw <= Capacity) &&(cp > reached)){
      reached = cp;
      #pragma unroll
      for(int i = 0; i < arraySize; i++){
        best_bin[i] = th_bin[i];
      }
    }
  }
  else{
    int cw = 0;
    for(int i = def_div ; i < arraySize; i++){
      cw += coefs[i] * th_bin[i];
    }
    if (cw > Capacity) forward = false;
    cw = 0;
    float cp = 0;
    int nw = 0;
    int np = 0;
    #pragma unroll
    for(int i = def_div;i < arraySize;i++){
      np = th_bin[i]!=-1? th_bin[i] * coefs[i+arraySize]:coefs[i+arraySize];
      nw = th_bin[i]!=-1? th_bin[i] * coefs[i]: coefs[i];
      if(cw+nw <= Capacity){
        cw += nw;
        cp += np;
      }
      else{
        cp+=np*(Capacity-cw)/nw;
        break;
      }
    }
    int b = cp;
    if (b <= reached){
      forward = false;
    }
  }
  if(forward){if(h<arraySize-1){h++;}
              }
  }

sh_maxs[threadIdx.x] += reached;




__syncthreads();
//reduction on block
  for (uint offset = blockDim.x >> 1; offset >= 1; offset >>= 1)
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
  sh_sum_dev[blockIdx.x] = sh_maxs[0];
  str_num_dev[blockIdx.x] = indices[0];
  }
  if(blockIdx.x*blockDim.x+threadIdx.x == indices[0]){
    #pragma unroll
    for(int i = 0; i< arraySize;i++){
      global_mem_bin[blockIdx.x*arraySize + i] = best_bin[i];
  }
  }
  __syncthreads();

}

__global__ void
reduction_max (float *s, long int *str_num_dev,int* global_mem_bin)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int sdata[threads_per_block*2];
  sdata[threadIdx.x] = s[ID];
  sdata[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads ();
  // do reduction in shared mem
  for (uint s = blockDim.x >>1; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
	{
	  if (sdata[threadIdx.x] < sdata[threadIdx.x + s])
	    {
	      sdata[threadIdx.x] = sdata[threadIdx.x + s];
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

            #pragma unroll
            for(int i = 0; i < arraySize;i++){
              global_mem_bin[i] = global_mem_bin[(sdata[threads_per_block]-arraySize)/threads_per_block*arraySize + i];
          }
		}

}


__global__ void
which_string (long int a, int *view_dev,int* global_mem_bin)
{
  view_dev[threadIdx.x] = global_mem_bin[blockIdx.x*arraySize + threadIdx.x];
}


void quickSortR(float* a,float* b, long N) {
// На входе - массив a[], a[N] - его последний элемент.

    long i = 0, j = N;      // поставить указатели на исходные места
    float temp, p;

    p = a[ N>>1 ];      // центральный элемент

    // процедура разделения
    do {
        while ( a[i] > p ) i++;
        while ( a[j] < p ) j--;

        if (i <= j) {
            temp = a[i]; a[i] = a[j]; a[j] = temp;
            temp = b[i]; b[i] = b[j]; b[j] = temp;
            temp = b[i+arraySize]; b[i+arraySize] = b[j+arraySize]; b[j+arraySize] = temp;
            i++; j--;
        }
    } while ( i<=j );

    // рекурсивные вызовы, если есть, что сортировать
    if ( j > 0 ) quickSortR(a,b, j);
    if ( N > i ) quickSortR(a+i,b+i, N-i);
}



    int main(){

      long int strSize_b = pow (2, arraySize);
      int num_of_blocks = strSize_b / threads_per_block;
      float *Sum = new float[32];	// = { 0 };
      float *sh_sum_dev;
      //float weight[31] ={ 5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107,115 };
      //float values[31] ={ 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313,321 };
      float dev_coefs[62] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107,115, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313,321 };
      //float dev_coefs[60] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313 };
      //float dev_coefs[58] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305 };

      //float *values_dev;
      long int *str_num_dev;
      long int *str_num = new long int[1];
      float N_of_rep;
      N_of_rep = num_of_blocks/max_blocks;
      cout <<"N of items "<<arraySize<<"\n";
      cout<<"N of blocks "<<num_of_blocks<<"\n";
      cout<<"strSize_b = "<<strSize_b<<"\n";
      cout<<"num_of_blocks / threads_per_block = "<<num_of_blocks / threads_per_block<<"\n";
      cout<<"N of repeats = "<<N_of_rep<<"\n";
      cout<<"sing param = "<<num_of_blocks/N_of_rep<<" _ "<< threads_per_block<<"\n";
      cout<<"red param "<<num_of_blocks / threads_per_block<<"  ,  "<<strSize_b/num_of_blocks<<"\n";

      float* additional_array = new float[arraySize];
      for(int i = 0; i < arraySize;i++){
      additional_array[i] = dev_coefs[i+arraySize]/dev_coefs[i];
      }

      quickSortR(additional_array,dev_coefs,arraySize-1);

      for(int i = 0;i<arraySize*2;i++){dev_coefs[i] = 2;}

      std::chrono::time_point<std::chrono:: high_resolution_clock> start, end;
          start = std::chrono::high_resolution_clock::now();

      int* bdevX;
      cudaMalloc ((void **) &bdevX, arraySize * sizeof (int));
      int* global_mem_bin;
      cudaMalloc ((void **) &global_mem_bin, max_blocks*arraySize * sizeof (int));





      cudaMalloc ((void **) &sh_sum_dev,  num_of_blocks * sizeof (float));
      cudaMalloc ((void **) &str_num_dev, num_of_blocks * sizeof (float));
      cudaMemcpyToSymbol (coefs, dev_coefs, 2*arraySize * sizeof (float));

    //int sing_blocks = num_of_blocks/N_of_rep;

            //for(int i = 0;i<N_of_rep;i++){
              //cout<<i;
      single_thread <<< max_blocks, threads_per_block >>> (sh_sum_dev, str_num_dev, num_of_blocks,bdevX,global_mem_bin);
                 //}



reduction_max<<<1,max_blocks>>>(sh_sum_dev,str_num_dev,global_mem_bin);
int* suda = new int[arraySize];
      cudaMemcpy (Sum, sh_sum_dev, sizeof (int), cudaMemcpyDeviceToHost);
      cudaMemcpy (str_num, str_num_dev, sizeof (long int), cudaMemcpyDeviceToHost);
      cudaMemcpy (suda, global_mem_bin, arraySize*sizeof (int), cudaMemcpyDeviceToHost);


      end = std::chrono:: high_resolution_clock::now();

          int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                                   (end-start).count();
          std::time_t end_time = std::chrono::system_clock::to_time_t(end);

          std::cout<< "Время выполнения: " << elapsed_seconds << "microseconds\n";

      cout << "Acheived maximal sum = " << Sum[0] << "\n";
      cout<<str_num[0]<<"\n";
        int *view = new int[arraySize];
        //int *view_dev;
        //cudaMalloc ((void **) &view_dev, arraySize * sizeof (int));
        //which_string <<< 1, arraySize >>> (str_num[0], view_dev,global_mem_bin);
        //cudaMemcpy (view, view_dev, arraySize * sizeof (int),
      	  //    cudaMemcpyDeviceToHost);
        for (int i = 0; i < arraySize; i++)
          {
            cout << suda[i] << " ";
          } cout << "\n";
        //check
        int checksum = 0;
        for (int i = 0; i < arraySize; i++)
          {
            checksum += dev_coefs[i+arraySize] * suda[i];
          }
        cout << "Validation sum = " << checksum << "\n";
        checksum = 0;
        for (int i = 0; i < arraySize; i++)
          {
            checksum += dev_coefs[i] * suda[i];
          } cout << "Weight = " << checksum << "\n";



        cudaFree(coefs);
        cudaFree (sh_sum_dev);
        cudaFree (str_num_dev);
        cudaFree(bdevX);
        cudaFree(global_mem_bin);

        delete [] Sum;
        delete [] str_num;
        delete [] additional_array;

        cout<<"Проверка. CPU version:\n";
        int *X = new int[arraySize];
        int *bestX = new int[arraySize];
        for(int i = 0; i < arraySize; i++){
          X[i] = -1;
          bestX[i] = 0;
        }
        int curr_sum = 0;
        int reached_max = 0;

        float *cpu_bin = new float[arraySize];

        start = std::chrono::high_resolution_clock::now();

        int h = 0;
        int k = h;//def_div;
        long int ns = 0;
        bool forward;
        while(h-k!=-1){
          ns++;
          forward = true;
          if(X[h]==-1){
            X[h]=1;
          }else{
          if(X[h]==1){
            X[h]=0;
          }else{
          if(X[h]==0){
            X[h]=-1;
            h--;
            forward=false;
          }
        }
        }
          if(h==arraySize-1){
            int cw = 0;
            int cp = 0;
            for(int i = k;i<arraySize;i++){
              cp += dev_coefs[i+arraySize]*X[i];
              cw += dev_coefs[i]*X[i];
            }
            if((cw <= W) &&(cp > reached_max)){
              reached_max = cp;
              for(int i = k; i < arraySize; i++){
                bestX[i] = X[i];
              }
            }
          }
          else{
            int cw = 0;
            for(int i = k ; i < arraySize; i++){
              cw += dev_coefs[i]*X[i];
            }
            if (cw > W) forward = false;
            cw = 0;
            float cp = 0;
            int nw = 0;
            int np = 0;
            for(int i = k;i<arraySize;i++){
              np = X[i]!=-1? X[i] * dev_coefs[i+arraySize]:dev_coefs[i+arraySize];
              nw = X[i]!=-1? X[i] * dev_coefs[i]: dev_coefs[i];
              if(cw+nw <= W){
                cw += nw;
                cp += np;
              }
              else{
                cp+=np*(W-cw)/nw;
                break;
              }
            }
            int b = cp;
            if (b <= reached_max){
              forward = false;
            }
          }
          if(forward){if(h<arraySize-1){h++;}}
          }


          end = std::chrono:: high_resolution_clock::now();

              elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                                       (end-start).count();
               end_time = std::chrono::system_clock::to_time_t(end);

          std::cout<< "Время выполнения: " << elapsed_seconds << "microseconds\n";

        cout<<"MAX = "<<reached_max<<"\n";
        for(int k = 0 ; k < arraySize;k++){
        cout<<bestX[k];
        curr_sum += bestX[k]*dev_coefs[k+arraySize];
        }cout<<"\nЧисло итераций = "<<ns<<"\n";

delete [] suda;

return 0;
}
