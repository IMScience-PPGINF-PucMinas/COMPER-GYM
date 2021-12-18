#export DISPLAY="`grep nameserver /etc/resolv.conf | sed 's/nameserver //'`:0"
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
import gym
mujoco_taks=['Hopper-v2','HalfCheetah-v2','Walker2d-v2','Swimmer-v2','Humanoid-v2','Ant-v2']


def run(task,max_steps=1000,render=False):
    env = gym.make(task)
    print("\nTask------",task)
    print("Action Space:", env.action_space)
    #print("Observation Space: ",env.observation_space)
    print ("Observation size",env.observation_space)
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    done = False
    step =0
    st = env.reset()
    print("st_shape: ",st.shape)
    #print(st)
    while(step<=1 ):#and not done):
        if(render):    
            env.render()
        action = env.action_space.sample()
        print("Action:",action.shape)
        st, r, done, _ =env.step(action) # take a random action
        
        #print(st)
        step+=1
               
    env.close()
    print("steps",step)

for i in range(len(mujoco_taks)):
    run(mujoco_taks[i],max_steps=1 )