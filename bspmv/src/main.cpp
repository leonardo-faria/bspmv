#include <string.h>

#include <stdlib.h>
#include <stdio.h>
#include "tests/testDir.h"
#include "tests/benchmarkBlock.h"


int main(int argc, char **argv) {
	testMatrix("/media/leonardo/EA1696FC1696C949/matrices from linux/data 32768 cube Elements penalty 10 P2 basis.mtx", "/media/leonardo/EA1696FC1696C949/matrices from linux/data 32768 cube Elements penalty 10 P2 basis.mtx");
	return 1;
	if (argc == 2)
		testDir(argv[2]);
	if (argc == 3) {
		if (strcmp(argv[2], "-f") == 0) {
			testMatrix(argv[1], argv[1]);
		}
		if (strcmp(argv[1], "-f") == 0)
			testMatrix(argv[2], argv[2]);
	} else
		printf("Usage:\nbspmv <dir-name>\nbspmv -f <filename>\n");

}
