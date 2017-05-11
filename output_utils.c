#include <assert.h>
#include <stdio.h>
#include "rank.h"
#include "cudaRiver.h"

static void get_oop_action_lists(hand_t *range, char *strat, int n,
        hand_t *check_call, hand_t *check_fold, hand_t *bet,
        int *num_check_call, int *num_check_fold, int *num_bet) {
    int check_call_idx = 0, check_fold_idx = 0, bet_idx = 0;
    for (int i = 0; i < n; i++) {
        hand_t hand = range[i];
        switch (strat[i]) {
            case CHECK_CALL:
                check_call[check_call_idx++] = hand;
                break;
            case CHECK_FOLD:
                check_fold[check_fold_idx++] = hand;
                break;
            case BET:
                bet[bet_idx++] = hand;
                break;
            default:
                assert(0);
        }
    }

    *num_check_call = check_call_idx;
    *num_check_fold = check_fold_idx;
    *num_bet = bet_idx;
}

static void print_hand(hand_t *hand) {
    char card_str[CARD_STR_LEN];
    for (int i = 0; i < HAND_SIZE; i++) {
        card_to_str(hand->cards[i], card_str);
        printf("%s", card_str);
    }
}

void output_human_readable(hand_t *range, char *strat, int n) {
    char card_str[CARD_STR_LEN];

    hand_t check_call[n], check_fold[n], bet[n];
    int num_check_call, num_check_fold, num_bet;

    get_oop_action_lists(range, strat, n, check_call, check_fold, bet,
            &num_check_call, &num_check_fold, &num_bet);

    printf("\nCheck-call hands:\n");
    for (int i = 0; i < num_check_call; i++) {
        hand_t hand = check_call[i];
        print_hand(&hand);

        printf(",\n");
    }

    printf("\nCheck-fold hands:\n");
    for (int i = 0; i < num_check_fold; i++) {
        hand_t hand = check_fold[i];
        print_hand(&hand);

        printf(",\n");
    }


    printf("\nBet hands:\n");
    for (int i = 0; i < num_bet; i++) {
        hand_t hand = bet[i];
        print_hand(&hand);

        printf(",\n");
    }
}
