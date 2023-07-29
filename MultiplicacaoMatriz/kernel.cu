#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include "device_launch_parameters.h"


float* a, * b, * c; //host variables

//Incluindo modificação na branch

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    
    //Calcula o indice da linha do elemento P e M
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
 
    //Calcula o indice da coluna do elemento P e N
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;

        //Cada thread calcula um elemento do bloco da submatriz
        for (int k = 0; k < Width; k++) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }

        P[Row * Width + Col] = Pvalue;
    }
}

int main()
{
    cudaDeviceReset();

    float* d_a, * d_b, * d_c; //device variables

    int n = 16;
    int size = n * n * sizeof(float);
    dim3 dimGrid((n-1)/16 + 1, (n - 1) / 16 + 1, 1);
    dim3 dimBlock(16, 16, 1);


    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
   
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    for (int i = 0; i < n*n; i++)
        a[i] = 2, b[i] = 2;

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    
    MatrixMulKernel << <dimGrid,dimBlock >> > (d_a, d_b, d_c, n);
    cudaDeviceSynchronize;

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //   for (int i = 0; i < n; i++)
        printf("%f ", c[2]);

    cudaFree(d_a), cudaFree(d_b), cudaFree(d_c);

    return 0;
}
