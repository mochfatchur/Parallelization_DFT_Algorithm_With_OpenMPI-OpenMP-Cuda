#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#define MAX_N 512
#ifndef M_PI
 # define M_PI		3.14159265358979323846
#endif

struct Matrix {
    int size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int size;
    cuDoubleComplex mat[MAX_N][MAX_N];
};

__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}



__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// __global__ void sum_elements(struct FreqMatrix *mat, cuDoubleComplex *sum)
__global__ void sum_elements(struct Matrix *source , struct FreqMatrix *mat, cuDoubleComplex *sum) {
    // definisi row dan column
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;


    // dft untuk tiap row dan col tertentu
    cuDoubleComplex element = make_cuDoubleComplex(0.0, 0.0);
    for (int m = 0; m < source->size; m++) {
        for (int n = 0; n < source->size; n++) {
            double arg = (row * m / (double)source->size) + (col * n / (double)source->size);
            cuDoubleComplex exponent = cuCexp(make_cuDoubleComplex(0.0, -2.0 * M_PI * arg));
            cuDoubleComplex mat_elem = make_cuDoubleComplex(source->mat[m][n], 0.0);
            element = cuCadd(element, cuCmul(mat_elem, exponent));
        }
    }

    // dijumlahkan dengan atomic add untuk mencapai sum Total
    mat->mat[row][col] = cuCdiv(element, make_cuDoubleComplex(mat->size * mat->size, 0.0));
    atomicAddDouble(&sum->x, mat->mat[row][col].x);
    atomicAddDouble(&sum->y, mat->mat[row][col].y);

}

int main(void) {
    struct Matrix source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    // copy sum
    cuDoubleComplex *d_sum;
    // copy matrix freq_domain
    struct FreqMatrix *d_freq_domain;
    // copy source
    struct Matrix *d_matrix;

    // alokasi memori untuk device copy dari sum dan copy dari matrix freq_domain
    cudaMalloc((void **)&d_sum, sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_freq_domain, sizeof(struct FreqMatrix));
    cudaMalloc((void **)&d_matrix, sizeof(struct Matrix));


    // copy sum dan matrix freq_domain ke devices/GPU
    cudaError_t error = cudaMemcpy(d_sum, &sum, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    cudaError_t error1 = cudaMemcpy(d_freq_domain, &freq_domain, sizeof(struct FreqMatrix), cudaMemcpyHostToDevice);
    if (error1 != cudaSuccess){
        printf("%s\n", cudaGetErrorString(error1));
    }

    // matrix source
    cudaMemcpy(d_matrix, &source, sizeof(struct Matrix), cudaMemcpyHostToDevice);

    // init block and grid
    dim3 block(16, 16);    // dimensi block 16 * 16 * 1
    dim3 grid((int) ceil (source.size/block.x), (int)ceil(source.size/block.y));  // dimensi grid (2,2) 32/16 32/16 => 2 * 64/16 * 64/16

    // GPU
    // sum_elements<<<grid,block>>>(d_freq_domain, d_sum);
    sum_elements<<<grid,block>>>(d_matrix, d_freq_domain, d_sum);

    cudaError_t error0 = cudaGetLastError();
    if (error0 != cudaSuccess){
        printf("Cuda error 0 : %s\n", cudaGetErrorString(error0));
    }

    // mengambil hasil sum dari GPU ke host
    cudaError_t error2 = cudaMemcpy(&sum, d_sum, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (error2 != cudaSuccess){
        printf("sukses3 sum \n");
    }
    
    // mengambil hasil freq_domain dari devices ke host
    cudaMemcpy(&freq_domain, d_freq_domain, sizeof(struct FreqMatrix), cudaMemcpyDeviceToHost);

    // print hasil freq_domain
    for (int k = 0; k < freq_domain.size; k++) {
        for (int l = 0; l < freq_domain.size; l++) {
            printf("(%lf, %lf) ", freq_domain.mat[k][l].x, freq_domain.mat[k][l].y);
        }
        printf("\n");
    }

    // Mendapatkan hasil average
    sum = cuCdiv(sum, make_cuDoubleComplex(source.size, 0.0));
    printf("Average : (%lf, %lf)", cuCreal(sum), cuCimag(sum));

    // free alokasi
    cudaFree(d_sum);
    cudaFree(d_freq_domain);
    cudaFree(d_matrix);

    return 0;
}