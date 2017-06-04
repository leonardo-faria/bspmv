#include "cpu_coo.h"

void matrixvector(coo_sparse_matrix& coo, double* x, double* y) {
	for (int i = 0; i < coo.getRows(); ++i)
		y[i] = 0;
	for (int i = 0; i < coo.getNonz(); ++i){
		y[coo.getCpuIrp()[i]] += x[coo.getCpuJa()[i]] * coo.getCpuAs()[i];
	}
}

void test_csr_block(char* filename){

}
