"""           
     Transition: [st_1,a,r,st,q,done] -> size 14116.  
     Transition: [st_1,a,r,st,q,done] -> size 56452.                        
"""

class FrameTransitionTypes:
        NOT_DEFINED   = 0
        SINGLE_FRAMES = 1
        STAKED_FRAMES = 2

class FrameTransition:
        TYPE = FrameTransitionTypes.NOT_DEFINED        
        ST_W = 0
        ST_H = 0
        ST_L = 0    
        T_LENGTH = 0
        T_N_IDX  = 0
        T_IDX_ST_1 = [0,0]
        T_IDX_A    = 0
        T_IDX_R    = 0
        T_IDX_ST   = [0,0]
        T_IDX_Q    = 0
        T_IDX_DONE = 0

        