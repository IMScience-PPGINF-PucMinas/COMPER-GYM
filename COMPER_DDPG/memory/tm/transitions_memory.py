import numpy as np
from collections import defaultdict
from collections import deque
from memory.similarity.faiss_index import FaissIndex
from memory.tm.transitions_set import TSet
from memory.base.base_transitions_memory import BaseTransitionsMemory

class TransitionsMemory(BaseTransitionsMemory):
    def __init__(self,max_size,name,memory_dir='./'):
        self.faissidx ={}        
        self.t_set ={}
        self.tmax = max_size               
        
        super().__init__(name,memory_dir='./')       
        self.__initialize()

    def __initialize(self):
        self.__init_faiss()
        self.__init_tset()

    def __init_faiss(self):
        self.faissidx = FaissIndex(max_transitions=self.tmax,transition_length=self.tlen-2)

    def __init_tset(self):
        self.t_set = TSet(max_sets=self.tmax)

    def __len__(self):
        return self.t_set.len()   
    
    def __shape_transition(self,s_t_1,a_t_1,r_t,s_t,q_t,done):        
        t = np.array(s_t_1)
        t = np.insert(t,len(t),a_t_1)        
        t = np.insert(t,len(t),r_t)
        t = np.insert(t,len(t),s_t)
        t = np.insert(t,len(t),q_t)        
        t = np.insert(t,len(t),done)
        return np.array(t)
        
    def __exist_simillar_transition(self,t):        
        d,i = self.faissidx.get_sim_transition(t)
        exist = True if d[0][0]>=0.0 and d[0][0]<=0.0001 else False            
        return exist,i[0][0]   

    def add_transition(self,s_t_1,a_t_1,r_t,s_t,q_t,done):
        t = self.__shape_transition(s_t_1,a_t_1,r_t,s_t,q_t,done)
        _t = t[:-2]
        exist,similar_t_key = self.__exist_simillar_transition(_t)
        if(not exist):
            similar_t_key =self.faissidx.add_transition(_t)
        elif(exist):
            temp =0         
        self.t_set.add(t,similar_t_key)

    def load_transitions_batch_as_features_array(self,bsize=1000,include_done_singnal=False,normalizeFromMean=False):
        transitions= self.t_set.batch_tolist(bsize)
        transitions =transitions[:,:-1]#removing the done signals        
        return np.array(transitions)

    
    def sample_transitions_batch(self,batch_size=32):
        transitions=[]
        samp = self.t_set.random_samp(batch_size)        
        try:
            keys =list(samp.keys())
            for k in keys:                
                t = np.array(samp[k])
                t = t.reshape(1,self.tlen)
                if(len(transitions)==0):
                    transitions = t                    
                else:
                    transitions = np.concatenate((transitions,t),axis=0)
            return transitions        
        except Exception as e:          
           raise(e)
           
    def load_transitions_batch_as_features_array_grouped(self,start=0,bsize=1000,include_done_singnal=False,delete_from_memory=True):        
        samp = self.t_set.batch(bsize,delete_from_memory)
        return samp       
    
   
    