#ifndef _CUDA_RIVER_H
#define _CUDA_RIVER_H

#include "rank.h"

typedef struct GlobalConstants {
    //must be in cuda
    int *oopRanks;
    int oopSize;

    //must be in cuda
    int *ipRanks;
    int ipSize;
    int potSize;
    int betSize;
    int afterBetSize;
} GlobalConstants;

#define CHECK_CALL 0
#define CHECK_FOLD 1
#define BET 2
#define OOP_MOVES 3

#define IP_BET 0
#define IP_CHECK 1
#define IP_CALL 2
#define IP_FOLD 3

#define IP_MOVES 4

#define ITERATIONS_TO_PRINT 100000
#define NUM_STRATEGIES_PER_ITERATION 1000
#define MAX_THREADS 64 
#define MAX_BLOCKS 64

GlobalConstants *calcGlobalConsts(board_t board, hand_t *oopRange,
        int oopSize, hand_t *ipRange, int ipSize, int potSize, int betSize);

void calcMaxOopStrategy(char *bestStrat, int *stratVal, GlobalConstants *params);

void calcMaxIpStrategy(char *bestOopStrat, char *bestIpCheckStrat,
        char *bestIpBetStrat, GlobalConstants *params);
#endif /* _CUDA_RIVER_H */
