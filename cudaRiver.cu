#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>


extern "C" {
#include "cudaRiver.h"
}

/*struct GlobalConstants {
    //must be in cuda
    int *oopRanks;
    int oopSize;

    //must be in cuda
    int *ipRanks;
    int ipSize;
    int potSize;
    int betSize;
    int afterBetSize;
};*/

__constant__ GlobalConstants cuConsts;

#define CHECK_CALL 0
#define CHECK_FOLD 1
#define BET 2
#define OOP_MOVES 3

#define IP_MOVES 4

#define NUM_STRATEGIES_PER_ITERATION 100

char **cudaOopStrategies;
int *cudaOutput;
int *output;

__global__ void kernel_calculateValue(int handsPerThread,
        char **cudaOopStrategies, int *output) {
    int idx = threadIdx.x;
    int strategyIdx = blockIdx.x;

    char *strategy = cudaOopStrategies[strategyIdx];
    int check = 0;
    int bet = 0;
    int call = 0;
    int fold = 0;
    int cb_max = 0;
    int cf_max = 0;
    int ipRank, oopRank, oopMove, showdown, showPot, showBet;
    for (int i = idx; i < cuConsts.ipSize; i += handsPerThread) {
        ipRank = cuConsts.ipRanks[i];
        for (int j = 0; j < cuConsts.oopSize; j++) {
            oopRank = cuConsts.oopRanks[j];
            oopMove = strategy[j];
            showdown = ipRank > oopRank ? 1 : -1;
            showPot = ipRank > oopRank ? cuConsts.potSize : 0;
            showBet = showPot + (showdown * cuConsts.betSize);
            switch (oopMove) {
            case CHECK_CALL:
                check += showPot;
                bet += showBet;
            case CHECK_FOLD:
                check += showPot;
                bet += cuConsts.potSize;
            case BET:
                call += showBet;
                fold -= cuConsts.potSize;
            }
        }
        cb_max = check > bet ? check : bet;
        cf_max = call > fold ? call : fold;
        //could be a bottleneck
        atomicAdd(output + strategyIdx, cb_max + cf_max);
        check = 0;
        bet = 0;
        call = 0;
        fold = 0;
    }
}

extern "C"
GlobalConstants *calcGlobalConsts(card_t *board, card_t *oopRange,
        int oopSize, card_t *ipRange, int ipSize, int potSize, int betSize) {
    GlobalConstants *params = (GlobalConstants *) malloc(sizeof(GlobalConstants));
    int *oopRanks = (int *) malloc(oopSize * sizeof(int));
    int *ipRanks = (int *) malloc(ipSize * sizeof(int));
    //char buffer[64];
    //char handBuffer[6];
    for (int i = 0; i < oopSize; i++) {
        oopRanks[i] = rank_of(board, oopRange + (i*2));
        //int_to_hand(oopRanks[i], buffer);
        //card_to_str(oopRange[i*2], handBuffer);
        //card_to_str(oopRange[i*2+1], handBuffer + 3);
        //printf("Hand: %s%s, Score: %d, Rank: %s\n", handBuffer, handBuffer + 3, oopRanks[i], buffer);
    }
    for (int i = 0; i < ipSize; i++) {
        ipRanks[i] = rank_of(board, ipRange + (i*2));
        //int_to_hand(ipRanks[i], buffer);
        //card_to_str(ipRange[i*2], handBuffer);
        //card_to_str(ipRange[i*2+1], handBuffer + 3);
        //printf("Hand: %s%s, Score: %d, Rank: %s\n", handBuffer, handBuffer + 3, ipRanks[i], buffer);
    }
    cudaMalloc(&(params->oopRanks), sizeof(int) * oopSize);
    cudaMalloc(&(params->ipRanks), sizeof(int) * ipSize);

    cudaMemcpy(params->oopRanks, oopRanks, sizeof(int) * oopSize, cudaMemcpyHostToDevice);
    cudaMemcpy(params->ipRanks, ipRanks, sizeof(int) * ipSize, cudaMemcpyHostToDevice);

    params->oopSize = oopSize;
    params->ipSize = ipSize;
    params->potSize = potSize;
    params->betSize = betSize;
    params->afterBetSize = potSize + betSize;
    //printf("Done\n");
    return params;
}

void addOne(char *curStrategy, GlobalConstants *params) {
    for (int i = 0; i < params->oopSize - 1; i++) {
        curStrategy[i] = (curStrategy[i] + 1) % OOP_MOVES;
        if (curStrategy[i] != 0) break;
    }
}

//calculates the best strategy for the oop player along with the strategies value
extern "C"
void calcMaxStrategy(char *bestStrat, int *stratVal, GlobalConstants *params) {
    cudaMemcpyToSymbol(cuConsts, params, sizeof(GlobalConstants));


    char **oopStrategies =
        (char **) malloc(NUM_STRATEGIES_PER_ITERATION * sizeof(char *));
    cudaMalloc(&cudaOopStrategies, NUM_STRATEGIES_PER_ITERATION * sizeof(char *));

    for (int i = 0; i < NUM_STRATEGIES_PER_ITERATION; i++) {
        cudaMalloc(&oopStrategies[i], params->oopSize * sizeof(char));
    }
    cudaMemcpy(cudaOopStrategies, oopStrategies,
            NUM_STRATEGIES_PER_ITERATION * sizeof(char*), cudaMemcpyHostToDevice);

    int totalStrategies = 1;
    for (int i = 0; i < params->oopSize; i++) {
        totalStrategies *= OOP_MOVES;
    }

    output = (int *) malloc(NUM_STRATEGIES_PER_ITERATION * sizeof(int));
    cudaMalloc(&cudaOutput, NUM_STRATEGIES_PER_ITERATION * sizeof(int));

    char *curStrategy = (char *) malloc(params->oopSize * sizeof(char));
    memset(curStrategy, 0, params->oopSize * sizeof(char));

    int numThreads = 64 > params->ipSize ? params->ipSize : 64;

    int handsPerThread = params->ipSize / numThreads;
    char *minStrategy = (char *) malloc(params->oopSize * sizeof(char));
    int minFound = INT_MAX;

    //number of kernel invokations needed
    for (int i = 0; i < totalStrategies / NUM_STRATEGIES_PER_ITERATION; i++) {
        //strategies per kernel call
        for (int j = 0; j < NUM_STRATEGIES_PER_ITERATION; j++) {
            addOne(curStrategy, params);
            if (cudaMemcpy(oopStrategies[j], curStrategy, params->oopSize *
                        sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess) {
                printf("CudaMemcpy Failed\n");
                assert(0);
            }
        }
        cudaMemset(cudaOutput, 0, NUM_STRATEGIES_PER_ITERATION * sizeof(int));
        kernel_calculateValue<<<NUM_STRATEGIES_PER_ITERATION, numThreads>>>
            (handsPerThread, cudaOopStrategies, cudaOutput);
        if (cudaMemcpy(output, cudaOutput, NUM_STRATEGIES_PER_ITERATION * sizeof(int),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("CudaMemcpy Failed\n");
            assert(0);
        }

        //need to synchronize here
        int minIdx = -1;
        //output is the value to the ip strategy, so find the minimum
        for (int k = 0; k < NUM_STRATEGIES_PER_ITERATION; k++) {
            if (output[k] < minFound) {
                minIdx = k;
                minFound = output[k];
            }
        }
        if (minIdx >= 0 && cudaMemcpy(bestStrat, oopStrategies[minIdx],
                    params->oopSize * sizeof(char),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("CudaMemcpy Failed\n");
            assert(0);
        }
    }
    *stratVal = minFound;
}
