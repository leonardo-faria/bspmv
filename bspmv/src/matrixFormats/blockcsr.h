/*
 * blockcsr.h
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */
#include "sparsematrix.h"
#ifndef BLOCKCSR_H_
#define BLOCKCSR_H_

class block_csr : public sparse_matrix {
	unsigned int block_width;
	unsigned int block_height;
	unsigned int block_size;
	unsigned int size_irp;
	unsigned int size_ja;
	unsigned int size_as;
	unsigned int block_rows;
	unsigned int* cpu_irp;
	unsigned int* cpu_ja;
	double* cpu_as;
public:
	block_csr(sparse_matrix &s,unsigned int beh,unsigned int bew);
	virtual ~block_csr();
	virtual coo_sparse_matrix* to_coo(){return 0;};
	virtual void from_coo(coo_sparse_matrix coo) {
	}
	unsigned int getBlockHeight() const {
		return block_height;
	}

	unsigned int getBlockSize() const {
		return block_size;
	}

	unsigned int getBlockWidth() const {
		return block_width;
	}

	double* getCpuAs() const {
		return cpu_as;
	}

	unsigned int* getCpuIrp() const {
		return cpu_irp;
	}

	unsigned int* getCpuJa() const {
		return cpu_ja;
	}

	unsigned int getSizeAs() const {
		return size_as;
	}

	unsigned int getSizeIrp() const {
		return size_irp;
	}

	unsigned int getSizeJa() const {
		return size_ja;
	}

	unsigned int getBlockRows() const {
		return block_rows;
	}

	;
};

#endif /* BLOCKCSR_H_ */
