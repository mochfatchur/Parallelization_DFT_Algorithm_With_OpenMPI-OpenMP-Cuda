#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512
#define ROOT 0

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

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

void Get_input(int my_rank, struct Matrix *m) {
    if (my_rank == 0) {
        readMatrix(m);
    }
    MPI_Bcast(&(m->size), 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < m->size; i++) {
        MPI_Bcast(&(m->mat[i]), m->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}


int main(void) {
    // inisialisasi matrix
    struct Matrix     source;
    struct FreqMatrix freq_domain;
    double complex local_sum = 0, total_sum = 0;

    // inisialisasi MPI
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Get Input
    Get_input(world_rank, &source);
    freq_domain.size = source.size;
    
    // satu proses menangani sejumlah (size_matrix / world_size) baris
    // satu prosesnya menangani baris row=rank + k * world_size
    // (misal pada size_matrix=32, world_size=4 proses 0 menangani baris 0, 4, 8, 12, 16 dll)
    // (misal pada size_matrix=32, world_size=4 proses 1 menangani baris 1, 5, 9, 13, 17 dll)
    for (int i = 0; i < source.size; i++)
    {
        if (i % world_size == world_rank) {
            for (int l = 0; l < source.size; l++) {
                freq_domain.mat[i][l] = dft(&source, i, l);
                double complex el = freq_domain.mat[i][l];
                // printf("(%lf, %lf) ", creal(el), cimag(el));
                local_sum += el;
            }
        }
    }

    // Jika rank selain 0, akan mengirim local_sum complex ke rank 0,
    // bilangan realnya dikirim dengan tag=0, dan bilangan imaginer dikirim dengan tag=1
    // lalu pada rank 0 akan dijumlahkan semua dari hasil prosesnya sendiri maupun dari rank lain
    if (world_rank != 0) {
        double real = creal(local_sum);
        double imaginer = cimag(local_sum);
        MPI_Send(&real, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&imaginer, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    else {
        total_sum += local_sum;
        for (int i = 1; i < world_size; i++) {
            double real, imaginer;
            MPI_Recv(&real, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&imaginer, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double complex received = real + imaginer*I;
            total_sum += received;
        }
        total_sum /= source.size;
        
        for (int l = 0; l < source.size; l++) {
            double complex el = freq_domain.mat[0][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
        }
        printf("\nAverage : (%lf, %lf)", creal(total_sum), cimag(total_sum));
    }

    // finalize
    MPI_Finalize();
    return 0;
}