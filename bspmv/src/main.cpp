#include "tests/testCsr.h"
#include "tests/testEll.h"
int main(int argc, char **argv) {

	if (argc == 1) {
//		testCsrforMatrix("matrices/test.mtx", 32);
//		testEllforMatrix("matrices/test.mtx", 32);
				testCsrforMatrix("/media/leonardo/EA1696FC1696C949/matrices\ from\ linux/data\ 512\ cube\ Elements\ penalty\ 10\ P1\ basis.mtx",32);
		testEllforMatrix("/media/leonardo/EA1696FC1696C949/matrices\ from\ linux/data\ 512\ cube\ Elements\ penalty\ 10\ P1\ basis.mtx",32);
	} else {
//		b_spmv(argv[1], 32);

	}
}

//tenho d por as cenas a chamar corretamenete
//forma mais facil é fazer api de cuda q recebe menos argumentos
