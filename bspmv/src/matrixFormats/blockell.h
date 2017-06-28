/*
 * blockell.h
 *
 *  Created on: 05/05/2017
 *      Author: leonardo
 */

#include "sparsematrix.h"
#ifndef BLOCKELL_H_
#define BLOCKELL_H_

class block_ell : public sparse_matrix {
	unsigned int block_width;
	unsigned int block_height;
	unsigned int block_size;
	unsigned int block_rows;
	unsigned int max_blocks;
	unsigned int size_ja;
	unsigned int size_as;
	unsigned int * cpu_ja;
	double* cpu_as;
public:
	block_ell(sparse_matrix & s,unsigned int beh,unsigned int bew);
	virtual ~block_ell();
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

	unsigned int* getCpuJa() const {
		return cpu_ja;
	}

	unsigned int getMaxBlocks() const {
		return max_blocks;
	}

	unsigned int getSizeAs() const {
		return size_as;
	}

	unsigned int getSizeJa() const {
		return size_ja;
	}

	unsigned int getBlockRows() const {
		return block_rows;
	}

	;
};

#endif /* BLOCKELL_H_ */
