#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cudaRiver.h"
#include "output_utils.h"

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
    GlobalConstants *params = calcGlobalConsts(board, oopRange, oopSize, ipRange, ipSize, potSize, betSize);

    char *bestOopStrategy = (char *) malloc(oopSize * sizeof(char));
    int stratVal;
    calcMaxStrategy(bestOopStrategy, &stratVal, params);

    char *bestIpCheckStrategy = (char *) malloc(ipSize * sizeof(char));
    char *bestIpBetStrategy = (char *) malloc(ipSize * sizeof(char));

    calcMaxIpStrategy(bestOopStrategy, bestIpCheckStrategy, bestIpBetStrategy, params);

    output_human_readable(oopRange, bestOopStrategy, oopSize,
            ipRange, bestIpCheckStrategy, bestIpBetStrategy, ipSize);

    clock_t end = clock();
    double time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Total time taken: %.2f seconds\n", time);
}
