OUTPUT_FOLDER = bin

all: serial parallel

parallel:
# TODO : Parallel compilation
	mpicc src/open-mpi/open-mpi.c -o $(OUTPUT_FOLDER)/mpi -lm

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm


# Compile
mpi: 
	mpicc src/open-mpi/open-mpi.c -o $(OUTPUT_FOLDER)/mpi -lm
mp:	
	gcc src/open-mp/open-mp.c --openmp -o $(OUTPUT_FOLDER)/mp -lm
cuda:
	nvcc src/cuda/cudarev.cu -o $(OUTPUT_FOLDER)/cuda

# Run EXEC
run-mpi: 
	time mpirun -n 5 ./bin/mpi <./test_case/32.txt> output_mpi_32.txt
run-mp: 
	time ./bin/mp < ./test_case/128.txt > output_mp_128.txt
run-serial: 
	time ./bin/serial < ./test_case/128.txt > output_serial.txt

run-cuda:
	time ./bin/cuda < ./test_case/32.txt > output_cuda.txt
