#include<stdio.h>
#include<cuda.h>

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

cudaMalloc((void**) &d_A, size);
cudaMalloc((void**) &d_B, size);

cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B,B, size, cudaMemcpyHostToDevice);
cudaMalloc((void**) &d_C, size);

vecAddKernel<<<ceil(n/256.0),256>>>(d_A, d_B, d_C, n);
cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
}


int main()
{
float *h_A= new float[5], *h_B= new float[5], *h_C = new float[5];
for(int i = 0; i < 5 ; i++)h_A[i]=1,h_B[i]=3;

vecAdd(h_A,h_B,h_C,5);

for(int i = 0 ; i< 5;i++) printf("%f\n", h_C[i]);
return 0;
}








