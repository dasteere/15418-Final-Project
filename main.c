#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "cudaRiver.h"
#include "output_utils.h"

#define CUDA

int main() {
    clock_t start = clock();
    int oopSize, ipSize, potSize, betSize;
    char card1[2], card2[2];
    scanf("%d %d %d %d\n", &oopSize, &ipSize, &potSize, &betSize);
    hand_t *oopRange = (hand_t *) malloc(oopSize * sizeof(hand_t));
    hand_t *ipRange = (hand_t *) malloc(ipSize * sizeof(hand_t));
    board_t board;
    for (int i = 0; i < BOARD_SIZE; i++) {
        scanf("%c%c,", card1, card1 +1);
        if (str_to_card(card1, &board.cards[i]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
    }
    scanf("\n---OOP RANGE---\n");
    for (int i = 0; i < oopSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &oopRange[i].cards[0]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &oopRange[i].cards[1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }
    scanf("---IP RANGE---\n");
    for (int i = 0; i < ipSize; i++) {
        scanf("%c%c,%c%c,\n", card1, card1 + 1, card2, card2 + 1);
        if (str_to_card(card1, &ipRange[i].cards[0]) < 0) {
            printf("Invalid input: %c%c\n", card1[0], card1[1]);
            return 0;
        }
        if (str_to_card(card2, &ipRange[i].cards[1]) < 0) {
            printf("Invalid input: %c%c\n", card2[0], card2[1]);
            return 0;
        }
    }

#ifdef CUDA
    char *bestOopStrategy = (char *) malloc(oopSize * sizeof(char));
    int stratVal;

    char *bestIpCheckStrategy = (char *) malloc(ipSize * sizeof(char));
    char *bestIpBetStrategy = (char *) malloc(ipSize * sizeof(char));

    GlobalConstants *params = calcGlobalConsts(board, oopRange, oopSize, ipRange, ipSize, potSize, betSize);
    calcMaxOopStrategy(bestOopStrategy, &stratVal, params);
    calcMaxIpStrategy(bestOopStrategy, bestIpCheckStrategy, bestIpBetStrategy, params);

    output_human_readable(oopRange, bestOopStrategy, oopSize,
            ipRange, bestIpCheckStrategy, bestIpBetStrategy, ipSize);
#else
    oop_action *oop_strat = (oop_action *)malloc(oopSize * sizeof(oop_action));
    assert(oop_strat);
    ip_action *ip_check_strat = (ip_action *)malloc(ipSize * sizeof(ip_action));
    assert(ip_check_strat);
    ip_action *ip_bet_strat = (ip_action *)malloc(ipSize * sizeof(ip_action));
    assert(ip_bet_strat);

    int *oop_ranks_arr = (int *) malloc(oopSize * sizeof(int));
    assert(oop_ranks_arr);
    int *ip_ranks_arr = (int *) malloc(ipSize * sizeof(int));
    assert(ip_ranks_arr);

    for (int i = 0; i < oopSize; i++) {
        oop_ranks_arr[i] = rank_of(&board, &oopRange[i]);
    }
    for (int i = 0; i < ipSize; i++) {
        ip_ranks_arr[i] = rank_of(&board, &ipRange[i]);
    }
    ranks_t oop_ranks;
    oop_ranks.n = oopSize;
    oop_ranks.r = oop_ranks_arr;

    ranks_t ip_ranks;
    ip_ranks.n = ipSize;
    ip_ranks.r = ip_ranks_arr;

    set_pot_size(potSize);
    set_bet_size(betSize);
    best_oop_strat(&oop_ranks, &ip_ranks, oop_strat);

    best_ip_strat(&oop_ranks, &ip_ranks, oop_strat,
            ip_check_strat, ip_bet_strat);

    output_human_readable_enum(oopRange, oop_strat, oopSize,
            ipRange, ip_check_strat, ip_bet_strat, ipSize);
#endif /* CUDA */

    clock_t end = clock();
    double time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Total time taken: %.2f seconds\n", time);
}
