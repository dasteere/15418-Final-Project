#include <cuda_runtime.h>
#include <driver_functions.h>

struct GlobalConstants {
    __constant__ int *oopRange;
    __constant__ int oopSize;

    __constant__ int *ipRange;
    __constant__ int ipSize;

    __constant__ int potSize;
    __constant__ int betSize;
    __constant__ int afterBetSize;
};

#define CHECK_CALL 1
#define CHECK_FOLD 2
#define BET 3
#define OOP_MOVES 3

#define CHECK 4
#define BET 5
#define CALL 6
#define FOLD 7
#define IP_MOVES 4

#define NUM_STRATEGIES_PER_ITERATION 100

char *oopStrategies[NUM_STRATEGIES_PER_ITERATION];
int output[NUM_STRATEGIES_PER_ITERATION];


__global__ void kernel_calculateValue(int handsPerThread, int *output) {
    int idx = threadIdx.x;
    int strategyIdx = blockIdx.x;

    char *strategy = oopStrategies[strategyIdx];
    int *out = output[strategyIdx];
    int check = 0;
    int bet = 0;
    int call = 0;
    int fold = 0;
    int cb_max = 0;
    int cf_max = 0;
    int ipRank, oopRank, oopMove, showdown, showPot, showBet;
    for (int i = idx; i < ipSize; i += handsPerThread) {
        int ipRank = ipRange[i];
        for (int j = 0; j < oopSize; j++) {
            int oopRank = oopRange[j];
            int oopMove = strategy[j];
            int showdown = ipRank > oopRank ? 1 : 0;
            showPot = showdown * potSize;
            showBet = showdown * afterBetSize;
            switch (oopMove) {
            case CHECK_CALL:
                check += showPot;
                bet += showBet;
            case CHECK_FOLD:
                check += showPot;
                bet += potSize;
            case BET:
                call += showdown * afterBetSize;
                fold -= potSize;
            }
        }
        cb_max = check > bet ? cc_check : cc_bet;
        cf_max = call > fold ? call : fold;
        //could be a bottleneck
        atomicAdd(output + strategyIdx, cb_max + cf_max);
        check = 0;
        bet = 0;
        call = 0;
        fold = 0;
    }
}

void calcMaxStrategy(char *bestStrat) {
    for (int i = 0; i < NUM_STRATEGIES_PER_ITERATION; i++) {
        oopStrategies[i] = (char *) malloc(oopSize * sizeof(char));
    }
    int totalStrategies = 1;
    for (int i = 0; i < oopSize; i++) {
        totalStrategies *= 3;
    }
    int totalStrategies =

}
