#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "rank.h"
#include "cpu-sequential.h"

#define CPU_SEQ_DBG

static int pot_size;
static int bet_size;

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) > (y) ? (y) : (x))

int compute_action_val(int oop_rank, int ip_rank, 
        oop_action oop_move, ip_action ip_move) {
    bool win_hand = ip_rank > oop_rank;

    switch (ip_move) {
        case I_CHECK:
            if (oop_move == OOP_BET) return 0;
            return win_hand ? pot_size : 0;
        case I_FOLD:
            return 0;
        case I_BET:
            if (oop_move == OOP_BET) return 0;
            if (oop_move == OOP_CHECK_CALL)
                return win_hand ? pot_size + bet_size : -1 * bet_size;
            return pot_size;
        case I_CALL:
            if (oop_move != OOP_BET) return 0;
            return win_hand ? pot_size + bet_size : -1 * bet_size;
    }
    assert(0);
}

/* Value of the optimum IP move for a specific IP hand
 * versus a given OOP strategy 
 * If ip_move_ptrs not NULL, place optimum moves there. */
int compute_ip_hand_val(ranks_t *oop_ranks, oop_action *oop_strat, int ip_rank,
        ip_action *ip_check_move_ptr, ip_action *ip_bet_move_ptr) {
    int check = 0, bet = 0, call = 0, fold = 0;

    for (int i = 0; i < oop_ranks->n; i++) {
        int oop_rank = oop_ranks->r[i];
        oop_action oop_move = oop_strat[i];

        call += compute_action_val(oop_rank, ip_rank, 
                oop_move, I_CALL);
        fold += compute_action_val(oop_rank, ip_rank,
                oop_move, I_FOLD);
        check += compute_action_val(oop_rank, ip_rank,
                oop_move, I_CHECK);
        bet += compute_action_val(oop_rank, ip_rank,
                oop_move, I_BET);
    } 

    if (ip_bet_move_ptr) {
        if (call > fold) *ip_bet_move_ptr = I_CALL;
        else *ip_bet_move_ptr = I_FOLD;

        if (check > bet) *ip_check_move_ptr = I_CHECK;
        else *ip_check_move_ptr = I_BET;
    }
    
    return MAX(check, bet) + MAX(call, fold);
}

/* Value of the optimum IP strategy given an OOP strategy *
 * If ip_strats not NULL, put the optimal IP strats there 
 */
int compute_ip_val(ranks_t *oop_ranks, ranks_t *ip_ranks, 
        oop_action *oop_strat, 
        ip_action *ip_check_strat, ip_action *ip_bet_strat) {

    int val = 0;
    ip_action *ip_check_move = NULL;
    ip_action *ip_bet_move = NULL;

    for (int i = 0; i < ip_ranks->n; i++) {
        int ip_rank = ip_ranks->r[i];

        if (ip_check_strat && ip_bet_strat) {
            ip_check_move = &ip_check_strat[i];
            ip_bet_move = &ip_bet_strat[i];
        }
        
        val += compute_ip_hand_val(oop_ranks, oop_strat, ip_rank,
                ip_check_move, ip_bet_move);
    }
    return val;
}

/* Returns the number of carries performed */
int inc_oop_strat(oop_action *strat, int n) {
    for (int i = 0; i < n; i++) {
        strat[i] = (strat[i] + 1) % OOP_ACTIONS;
        if (strat[i] != 0) return i;
    }
    return n;
}

static void print_strat(oop_action *strat, int n) {
    for (int i = 0; i < n; i++) {
        printf("strat %d\n", strat[i]);
    }
    printf("\n\n");
}

int best_oop_strat(ranks_t *oop_ranks, ranks_t *ip_ranks, 
        oop_action *oop_strat) {
    int n = oop_ranks->n;

    oop_action *cur_strat = (oop_action *)malloc(sizeof(oop_action) * n);
    if (cur_strat == NULL) return -1;

    /* Zero out strat */
    for (int i = 0; i < n; i++) {
        cur_strat[i] = 0;
    }

    int min_max = INT_MAX;
    int iter = 0;
    do {
        int ip_val = compute_ip_val(oop_ranks, ip_ranks, cur_strat, 
                NULL, NULL);        
        //printf("strat val %d:\n", ip_val);
        //print_strat(cur_strat, n);
        if (ip_val < min_max) {
            min_max = ip_val;
            memcpy(oop_strat, cur_strat, sizeof(oop_action) * n);
        }
        iter++;

#ifdef CPU_SEQ_DBG
        if (iter % 100000 == 0)
            printf("Iter %d\n", iter);
#endif
    } while (inc_oop_strat(cur_strat, n) != n);

    free(cur_strat);
    return 0;
}

int best_ip_strat(ranks_t *oop_ranks, ranks_t *ip_ranks, oop_action *oop_strat,
        ip_action *ip_check_strat, ip_action *ip_bet_strat) {
    return compute_ip_val(oop_ranks, ip_ranks, oop_strat, 
                           ip_check_strat, ip_bet_strat);    
}

void set_pot_size(int pot) { pot_size = pot; }
void set_bet_size(int bet) { bet_size = bet; }
