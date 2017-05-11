#ifndef _OUTPUT_UTILS_H
#define _OUTPUT_UTILS_H

#include "cpu-sequential.h"

void output_human_readable(hand_t *oopRange, char *oopStrat, int oopSize,
        hand_t *ipRange, char *ipCheckStrat, char *ipBetStrat, int ipSize);

void output_human_readable_enum(hand_t *oopRange, oop_action *oopStrat, 
        int oopSize, hand_t *ipRange, ip_action *ipCheckStrat, 
        ip_action *ipBetStrat, int ipSize);


#endif /* _OUTPUT_UTILS_H */
