#include <stdio.h>
#include <stdlib.h>
#include "tests/testCsr.h"
#include "tests/testEll.h"
int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Invalid arguments\n");
		return 1;
	}
	testCsrforMatrix(argv[1]);
	testEllforMatrix(argv[1]);
}
