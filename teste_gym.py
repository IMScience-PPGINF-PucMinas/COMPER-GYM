#export DISPLAY="`grep nameserver /etc/resolv.conf | sed 's/nameserver //'`:0"
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
import gym
mujoco_taks=['Hopper-v2','HalfCheetah-v2','Walker2s-v2','Swimmer-v2','Humanoid-v2','Ant-v2']


def run(task,max_steps=1000,render=False):
    env = gym.make('Hopper-v2')
    print("Action Space:", env.action_space)
    print("Observation Space: ",env.observation_space)
    done = False
    step =0
    env.reset()
    while(step<=1000 and not done):
        if(render):    
            env.render()
        action = env.action_space.sample()
        st, r, done, _ =env.step(action) # take a random action        
    env.close()

for i in range(len(mujoco_taks)):
    run(mujoco_taks[i],1)