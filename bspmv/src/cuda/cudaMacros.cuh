/*
 * cudaMacros.h
 *
 *  Created on: 12/06/2017
 *      Author: leonardo
 */

#ifndef CUDAMACROS_H_
#define CUDAMACROS_H_

#define TRIES 5

#define TEST_CUDA 1
#if TEST_CUDA
#define CHECK_CUDA_ERROR(function)\
	function;\
	error = cudaGetLastError();\
	if(error != cudaSuccess)\
	{\
	exit(1);\
	}
#else
#define CHECK_CUDA_ERROR(function)\
	function;
#endif

#define SUM_POSITIONS_3(offset) \
	{\
		sdata[tid_0] += sdata[tid_0 + offset*3];\
		sdata[tid_1] += sdata[tid_1 + offset*3];\
		sdata[tid_2] += sdata[tid_2 + offset*3];\
	}

#define SUM_POSITIONS_H(BEH,OFFSET) \
		for(i=0;i<BEH;i++){\
			sdata[tid_0+i]+=sdata[tid_0+i+OFFSET*BEH];\
		}\

#define SUM_POSITIONS_4(offset)\
	{\
		sdata[tid_0] += sdata[tid_0 + offset*4];\
		sdata[tid_1] += sdata[tid_1 + offset*4];\
		sdata[tid_2] += sdata[tid_2 + offset*4];\
		sdata[tid_3] += sdata[tid_3 + offset*4];\
	}

#define SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS)\
	switch (BLOCKSIZE) {\
		case 1:\
			OPERATION<1,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 2:\
			OPERATION<2,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 4:\
			OPERATION<4,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 8:\
			OPERATION<8,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 16:\
			OPERATION<16,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 32:\
			OPERATION<32,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 64:\
			OPERATION<64,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 128:\
			OPERATION<128,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 256:\
			OPERATION<256,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 512:\
			OPERATION<512,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		case 1024:\
			OPERATION<1024,BEH,BEW><<< DIMGRID, DIMBLOCK, SMEMSIZE>>>ARGS;\
			break;\
		default:\
				exit(1);\
	}



#define SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,BEH,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS)\
	switch (BEW) {\
		case 1:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,1,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 2:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,2,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 3:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,3,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 4:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,4,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 5:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,5,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 6:\
			SWITCH_BLOCK_SIZE(BLOCKSIZE,BEH,6,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		default:\
			exit(1);\
	}

#define SWITCH_BLOCKENTRY_HEIGHT(BLOCKSIZE,BEH,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS)\
	switch (BEH) {\
		case 1:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,1,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 2:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,2,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 3:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,3,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 4:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,4,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 5:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,5,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		case 6:\
			SWITCH_BLOCKENTRY_WIDTH(BLOCKSIZE,6,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS);\
			break;\
		default:\
			exit(1);\
	}


#define SWITCH_BLOCKENTRY_SIZE_AND_CUDA_BLOCK_SIZE(BLOCKSIZE,BEH,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS)\
	 SWITCH_BLOCKENTRY_HEIGHT(BLOCKSIZE,BEH,BEW,OPERATION, DIMGRID, DIMBLOCK, SMEMSIZE,ARGS)
#endif /* CUDAMACROS_H_ */
