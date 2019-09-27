#include<stdio.h>
#include<cuda.h>
#include<time.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
int i = (threadIdx.x + blockDim.x * blockIdx.x)*2;
if(i<n) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n)
{
int size = n * sizeof(float);
float *d_A, *d_B, *d_C ;
//Allocating memory on device
cudaMalloc((void**) &d_A, size);
cudaMalloc((void**) &d_B, size);
  
//Copy data from Host to Device
cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B,B, size, cudaMemcpyHostToDevice);
  
//Allocating memory for the output  
cudaMalloc((void**) &d_C, size);

vecAddKernel<<<ceil(n/256.0),256>>>(d_A, d_B, d_C, n);
//Copy data from Device to Host
cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

//Free memory in Device
cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
}


int main()
{
  int n;
  //Size of the vectors
  scanf("%d", &n);
  
  //Allocating memory on Host
  float *h_A= new float[n], *h_B= new float[n], *h_C = new float[n];
  
  //Initializing vectors with random values
  srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		h_A[i] = rand(); h_B[i] = rand();
	}
  
  vecAdd(h_A,h_B,h_C,n);

  for(int i = 0 ; i< n ;i++) printf("%f\n", h_C[i]);
  
return 0;
}








