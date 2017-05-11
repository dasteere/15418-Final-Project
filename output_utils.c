#include <assert.h>
#include <stdio.h>
#include "rank.h"
#include "cudaRiver.h"
#include "output_utils.h"

static void get_ip_bet_action_lists(hand_t *range, char *strat, int n,
        hand_t *call, hand_t *fold, int *num_call, int *num_fold) {
    int call_idx = 0, fold_idx = 0;
    for (int i = 0; i < n; i++) {
        hand_t hand = range[i];
        switch (strat[i]) {
            case IP_CALL:
                call[call_idx++] = hand;
                break;
            case IP_FOLD:
                fold[fold_idx++] = hand;
                break;
            default:
                assert(0);
        }
    }

    *num_call = call_idx;
    *num_fold = fold_idx;
}

static void get_ip_check_action_lists(hand_t *range, char *strat, int n,
        hand_t *check, hand_t *bet, int *num_check, int *num_bet) {
    int check_idx = 0, bet_idx = 0;
    for (int i = 0; i < n; i++) {
        hand_t hand = range[i];
        switch (strat[i]) {
            case IP_CHECK:
                check[check_idx++] = hand;
                break;
            case IP_BET:
                bet[bet_idx++] = hand;
                break;
            default:
                assert(0);
        }
    }

    *num_check = check_idx;
    *num_bet = bet_idx;
}

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

void output_human_readable(hand_t *oopRange, char *oopStrat, int oopSize,
        hand_t *ipRange, char *ipCheckStrat, char *ipBetStrat, int ipSize) {
    char card_str[CARD_STR_LEN];

    int max = oopSize > ipSize ? oopSize : ipSize;

    hand_t check_call[oopSize], check_fold[oopSize], bet[max];
    int num_check_call, num_check_fold, num_bet;

    get_oop_action_lists(oopRange, oopStrat, oopSize, check_call, check_fold, bet,
            &num_check_call, &num_check_fold, &num_bet);

    printf("OOP Player Strategy:\nCheck-call hands:\n");
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

    hand_t call[ipSize], fold[ipSize], check[ipSize];
    int num_call, num_fold, num_check;

    get_ip_check_action_lists(ipRange, ipCheckStrat, ipSize,
            check, bet, &num_check, &num_bet);
    get_ip_bet_action_lists(ipRange, ipBetStrat, ipSize, call,
            fold, &num_call, &num_fold);

    printf("IP Player Strategy\n");
    printf("If the OOP Player checks to you:\n");
    printf("Check hands:\n");
    for (int i = 0; i < num_check; i++) {
        hand_t hand = check[i];
        print_hand(&hand);
        printf(",\n");
    }

    printf("\nBet hands:\n");
    for (int i = 0; i < num_bet; i++) {
        hand_t hand = bet[i];
        print_hand(&hand);
        printf(",\n");
    }

    printf("\nIf the OOP Player bets to you:\n");
    printf("Call Hands:\n");
    for (int i = 0; i < num_call; i++) {
        hand_t hand = call[i];
        print_hand(&hand);
        printf(",\n");
    }

    printf("Fold Hands:\n");
    for (int i = 0; i < num_fold; i++) {
        hand_t hand = fold[i];
        print_hand(&hand);
        printf(",\n");
    }

}
