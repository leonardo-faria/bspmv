#include "tests/testCsr.h"
#include "tests/testEll.h"
#include  "tests/benchmarkBlock.h"
#include "matrixFormats/blockcsr.h"
#include "matrixFormats/sparsematrix.h"
#include <string.h>

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
	if (argc == 3)
		if (strcmp(argv[1], "-csr") == 0) {
			b_spmv_csr_ratio(argv[2]);
			for (int i = 1; i <= 1024; i *= 2) {

				printf("BS=%d,\n", i);

				b_spmv_csr(argv[2], i);
				printf("\n,\n,\n");
			}
		} else if (strcmp(argv[1], "-ell") == 0) {
			b_spmv_ell_ratio(argv[2]);
			for (int i = 1; i <= 1024; i *= 2) {

				printf("BS=%d,\n", i);
				b_spmv_ell(argv[2], i);

				printf("\n,\n,\n");
			}

		}
	if (argc == 2)
		if (strcmp(argv[1], "-csr") == 0)
			b_spmv_csr("matrices/test.mtx", 32);
		else if (strcmp(argv[1], "-ell") == 0)
			b_spmv_ell("matrices/test.mtx", 32);
	if (argc == 1) {
		testEllforMatrix("matrices/test.mtx", 32);
//		b_spmv_csr("matrices/test.mtx", 32);
//		b_spmv_ell("matrices/test.mtx", 32);
	}

	return 0;

}

//tenho d por as cenas a chamar corretamenete
//forma mais facil é fazer api de cuda q recebe menos argumentos
