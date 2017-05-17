/*
 * sparsematrix.h
 *
 *  Created on: 30/04/2017
 *      Author: leonardo
 */

#ifndef SPARSEMATRIX_H_
#define SPARSEMATRIX_H_
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "globals.h"
class coo_sparse_matrix;

class sparse_matrix {
protected:
	int nonzeros=0;
	int rows=0;
	int collumns=0;

public:
	sparse_matrix();
	virtual ~sparse_matrix() {
	}
	;
	virtual coo_sparse_matrix* to_coo()=0;
	virtual void from_coo(coo_sparse_matrix coo)=0;
	virtual void transp() {
		int temp;
		temp = rows;
		rows = collumns;
		collumns = temp;
	}
	;

	int getCols() const {
		return collumns;
	}

	int getNonz() const {
		return nonzeros;
	}

	int getRows() const {
		return rows;
	}

	static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
		if (err == cudaSuccess)
			return;
		std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
		exit(1);
	}
};

class coo_sparse_matrix: public sparse_matrix {
private:
	int* work_irp;
	int* work_ja;
	double* work_as;

	int* cpu_irp;
	int* cpu_ja;
	double* cpu_as;


	void BottomUpMerge(int iLeft, int iRight, int iEnd);
	void BottomUpMergeSort(int n);
	void CopyArray(int n);
public:
	virtual coo_sparse_matrix* to_coo();
	virtual void from_coo(coo_sparse_matrix coo);
	coo_sparse_matrix(char* filename);
	~coo_sparse_matrix();

	void parse_matrix(char* filename);

	double* getCpuAs() const {
		return cpu_as;
	}

	int* getCpuIrp() const {
		return cpu_irp;
	}

	int* getCpuJa() const {
		return cpu_ja;
	}
};

#endif /* SPARSEMATRIX_H_ */
