#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};


void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = 0.0;
    for (int m = 0; m < mat->size; m++) {
        for (int n = 0; n < mat->size; n++) {
            double complex arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element += mat->mat[m][n] * exponent;
        }
    }
    return element / (double) (mat->size*mat->size);
}

__global__ void sum_elements(struct FreqMatrix *mat, double complex *sum) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // posisi element matrix di bentuk row-major-order matrix
    int index = row * MAX_N + col;

    // jika row dan col masih berada pada jangkauan size matrix
    if(col < width && row < width){
        double complex el = mat.mat[index];
        *sum += el;
    }
}


int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    double complex sum = 0.0;
    // copy sum
    double complex *d_sum;
    // copy matrix freq_domain
    struct FreqMatrix d_freq_domain;

    // size dari copy sum dan copy matrix freq_domain
    int size_sum = MAX_N * sizeof(double);
    int size_mat = MAX_N * MAX_N * sizeof(struct FreqMatrix);

    // alokasi memori untuk device copy dari sum dan copy dari matrix freq_domain
    cudaMalloc((void **)&d_sum, size_sum);
    cudaMalloc((void **)&d_freq_domain, size_mat);

    // inialisasi freq_domain
    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = dft(&source, k, l);
    
    // copy sum dan matrix freq_domain ke devices/GPU
    cudaMemcpy(d_sum, &sum, size_sum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq_domain, freq_domain, size_mat, cudaMemcpyHostToDevice);

    // init block and grid
    dim3 block(512, 512);    // dimensi block 512 * 512 * 1
    dim3 grid((int) ceil (MAX_N/block.x), (int)ceil(MAX_N/block.y));  // dimensi grid

    // GPU
    sum_elements<<<grid,block>>>(d_freq_domain, d_sum);

    // mengambil hasil sum dari GPU ke host
    cudaMemcpy(&sum, d_sum, size_sum, cudaMemcpyDeviceToHost);

    sum /= source.size;
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));

    // free alokasi
    cudaFree(d_sum);
    cudaFree(d_freq_domain);

    return 0;
}