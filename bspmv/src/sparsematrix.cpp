/*
 * sparsematrix.cpp
 *
 *  Created on: 30/04/2017
 *      Author: leonardo
 */

#include "sparsematrix.h"

#include <stdio.h>
#include <string.h>
#include "mmio.h"
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
sparse_matrix::sparse_matrix(){}
coo_sparse_matrix::coo_sparse_matrix(char* filename) {
	parse_matrix(filename);

//	CUDA_CHECK_RETURN(cudaMalloc((void ** )&gpu_as, sizeof(double) * nonzeros));
//	CUDA_CHECK_RETURN(cudaMalloc((void ** )&gpu_ja, sizeof(int) * nonzeros));
//	CUDA_CHECK_RETURN(cudaMalloc((void ** )&gpu_irp, sizeof(int) * nonzeros));
//	cudaMemcpy(gpu_as,cpu_as, sizeof(double) * nonzeros,cudaMemcpyHostToDevice);
//	cudaMemcpy(gpu_irp,cpu_irp, sizeof(int) * nonzeros,cudaMemcpyHostToDevice);
//	cudaMemcpy(gpu_ja,cpu_ja, sizeof(int) * nonzeros,cudaMemcpyHostToDevice);
}

coo_sparse_matrix::~coo_sparse_matrix() {
	if (cpu_as)
		free(cpu_as);
	if (cpu_ja)
		free(cpu_ja);
	if (cpu_irp)
		free(cpu_irp);
//	if (gpu_as)
//		CUDA_CHECK_RETURN(cudaFree(gpu_as));
//	if (gpu_ja)
//		CUDA_CHECK_RETURN(cudaFree(gpu_ja));
//	if (gpu_irp)
//		CUDA_CHECK_RETURN(cudaFree(gpu_irp));
}

void coo_sparse_matrix::from_coo(coo_sparse_matrix coo) {

}

coo_sparse_matrix* coo_sparse_matrix::to_coo() {
	return this;
}

void coo_sparse_matrix::parse_matrix(char* filename) {
	int ret_code;
	MM_typecode matcode;
	FILE *f;
	if ((f = fopen(filename, "r")) == NULL)
		throw std::runtime_error("Could not open file " + std::string(filename) + ".\n");

	if (mm_read_banner(f, &matcode) != 0) {
		throw std::runtime_error("Could not process Matrix Market banner.\n");
	}
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) {
		throw std::runtime_error(((std::string) ("Sorry, this application does not support Market Market type: [" + (std::string) mm_typecode_to_str(matcode) + "%s]\n")).c_str());
		exit(1);
	}
	if ((ret_code = mm_read_mtx_crd_size(f, &rows, &collumns, &nonzeros)) != 0)
		throw std::runtime_error("Unable to read matrix crd size");
	if (mm_is_hermitian(matcode) || mm_is_symmetric(matcode) || mm_is_skew(matcode)) {
		cpu_irp = (int*) malloc(sizeof(int) * nonzeros * 2);
		cpu_ja = (int*) malloc(sizeof(int) * nonzeros * 2);
		cpu_as = (double*) malloc(sizeof(double) * nonzeros * 2);
	} else {
		cpu_irp = (int*) malloc(sizeof(int) * nonzeros);
		cpu_ja = (int*) malloc(sizeof(int) * nonzeros);
		cpu_as = (double*) malloc(sizeof(double) * nonzeros);

	}
	int temp_irp, temp_ja;
	double temp_as;
	int index = 0;
	for (int i = 0; i < nonzeros; i++) {
		fscanf(f, "%d %d %lg\n", &temp_irp, &temp_ja, &temp_as);
		cpu_irp[index] = temp_irp - 1;
		cpu_ja[index] = temp_ja - 1;
		cpu_as[index] = temp_as;
		index++;
		if ((mm_is_hermitian(matcode) || mm_is_symmetric(matcode) || mm_is_skew(matcode)) && temp_irp != temp_ja) {
			cpu_irp[index] = temp_ja - 1;
			cpu_ja[index] = temp_irp - 1;
			cpu_as[index] = temp_as;
			index++;
		}
	}
	if (mm_is_hermitian(matcode) || mm_is_symmetric(matcode) || mm_is_skew(matcode)) {
		nonzeros = index;
		cpu_irp = (int*) realloc(cpu_irp, sizeof(int) * nonzeros);
		cpu_ja = (int*) realloc(cpu_ja, sizeof(int) * nonzeros);
		cpu_as = (double*) realloc(cpu_as, sizeof(double) * nonzeros);
	}

	work_irp = (int*) malloc(sizeof(int) * nonzeros);
	work_ja = (int*) malloc(sizeof(int) * nonzeros);
	work_as = (double*) malloc(sizeof(double) * nonzeros);
	BottomUpMergeSort(nonzeros);
	free(work_irp);
	free(work_ja);
	free(work_as);
}

void coo_sparse_matrix::BottomUpMergeSort(int n) {
	// Each 1-element run in A is already "sorted".
	// Make successively longer sorted runs of length 2, 4, 8, 16... until whole array is sorted.
	for (int width = 1; width < n; width = 2 * width) {
		// Array A is full of runs of length width.
		for (int i = 0; i < n; i = i + 2 * width) {
			// Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
			// or copy A[i:n-1] to B[] ( if(i+width >= n) )
			BottomUpMerge(i, std::min(i + width, n), std::min(i + 2 * width, n));
		}
		// Now work array B is full of runs of length 2*width.
		// Copy array B to array A for next iteration.
		// A more efficient implementation would swap the roles of A and B.
		CopyArray(n);
		// Now array A is full of runs of length 2*width.
	}
}
void coo_sparse_matrix::BottomUpMerge(int iLeft, int iRight, int iEnd) {
	int i = iLeft, j = iRight;
	// While there are elements in the left or right runs...

	for (int k = iLeft; k < iEnd; k++) {
		// If left run head exists and is <= existing right run head.
		if (i < iRight && (j >= iEnd || cpu_irp[i] < cpu_irp[j] || (cpu_irp[i] == cpu_irp[j] && cpu_ja[i] <= cpu_ja[j]))) {
			work_as[k] = cpu_as[i];
			work_irp[k] = cpu_irp[i];
			work_ja[k] = cpu_ja[i];
			i = i + 1;
		} else {
			work_as[k] = cpu_as[j];
			work_irp[k] = cpu_irp[j];
			work_ja[k] = cpu_ja[j];
			j = j + 1;
		}
	}
}
void coo_sparse_matrix::CopyArray(int n) {
	memcpy(cpu_as, work_as, sizeof(double) * n);
	memcpy(cpu_irp, work_irp, sizeof(int) * n);
	memcpy(cpu_ja, work_ja, sizeof(int) * n);
}
