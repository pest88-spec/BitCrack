NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)

# Define build variables
CXX=g++
NVCC=nvcc
CUDA_INCLUDE=/usr/local/cuda/include
CUDA_MATH=../cudaMath
LIBDIR=../lib
INCLUDE=-I../util -I../AddressUtil -I../CmdParse -I../CryptoUtil -I../KeyFinderLib -I../CLKeySearchDevice -I../cudaMath -I../clUtil -I../cudaUtil -I../secp256k1lib -I../Logger -I../embedcl
CXXFLAGS=-O2 -std=c++11
NVCCFLAGS=-O2 -std=c++11 -I${CUDA_INCLUDE}

all:	cuda

cuda:
	@echo "Compiling CudaKeySearchDevice..."
	@echo "INCLUDE: ${INCLUDE}"
	@echo "CUDA_INCLUDE: ${CUDA_INCLUDE}"
	@echo "CUDA_MATH: ${CUDA_MATH}"
	@echo "LIBDIR: ${LIBDIR}"

	# Create lib directory if it doesn't exist
	mkdir -p ${LIBDIR}

	# Compile C++ files
	for file in ${CPPSRC} ; do\
		echo "Compiling $$file..."; \
		${CXX} -c $$file ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS};\
	done

	# Compile CUDA files
	for file in ${CUSRC} ; do\
		echo "Compiling $$file..."; \
		${NVCC} -c $$file -o $$file".o" ${NVCCFLAGS} -rdc=true ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
	done

	# Device link
	${NVCC} -dlink -o cuda_libs.o *.cu.o -lcudadevrt -lcudart

	# Create static library
	ar rvs ${LIBDIR}/lib$(NAME).a *.o
	@echo "CudaKeySearchDevice compilation completed successfully!"

clean:
	rm -f *.o *.cu.o
	rm -f *.a

.PHONY: all cuda clean