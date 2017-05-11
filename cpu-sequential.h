#ifndef _CPU_SEQUENTIAL_H
#define _CPU_SEQUENTIAL_H

typedef struct {
    int *r;
    int n;
} ranks_t;

#define OOP_ACTIONS 3
#define IP_ACTIONS 4

typedef enum _oop_action { OOP_CHECK_CALL=0, 
    OOP_CHECK_FOLD=1, OOP_BET=2 } oop_action;

typedef enum _ip_action { I_CALL=0, I_FOLD=1, 
    I_BET=2, I_CHECK=3 } ip_action;

int compute_action_val(int oop_rank, int ip_rank, 
        oop_action oop_move, ip_action ip_move);
        
/* Value of the optimum IP strategy for a specific IP hand
 * versus a given OOP strategy 
 * If ip_move_ptrs not NULL, place optimum moves there. */
int compute_ip_hand_val(ranks_t *oop_ranks, oop_action *oop_strat,
        int ip_rank, ip_action *ip_check_strat,
        ip_action *ip_bet_strat); 

/* Value of the optimum IP strategy given an OOP strategy 
 * If ip_strats not NULL, put the optimal IP strats there 
 */
int compute_ip_val(ranks_t *oop_ranks, ranks_t *ip_ranks, 
        oop_action *oop_strat, 
        ip_action *ip_check_strat, ip_action *ip_bet_strat); 

/* Returns the number of carries performed */
int inc_oop_strat(oop_action *strat, int n); 

int best_oop_strat(ranks_t *oop_ranks, ranks_t *ip_ranks, 
        oop_action *oop_strat); 

int best_ip_strat(ranks_t *oop_ranks, ranks_t *ip_ranks, oop_action *oop_strat,
        ip_action *ip_check_strat, ip_action *ip_bet_strat);

void set_pot_size(int pot);
void set_bet_size(int bet);

#endif /* CPU_SEQUENTIAL_H */
