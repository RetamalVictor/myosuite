import myosuite
import gym
import skvideo.io
import numpy as np
import os
import pickle


#Loading the policy
pth = '../myosuite/agents/baslines_NPG/'
policy = pth + "myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
pi = pickle.load(open(policy, 'rb'))
print("policy loaded")


#Defining the environmnent
env = gym.make('myoElbowPose1D6MExoRandom-v0')
env.reset()

#Test positions
AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
env.reset()
frames = []
for ep in range(len(AngleSequence)):
    print("Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep]))
    env.env.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]
    env.env.target_type = 'fixed'
    env.env.weight_range=(0,0)
    env.env.update_target()
    for _ in range(40):
        frame = env.sim.render(width=400, height=400,mode='offscreen', camera_name=None)
        frames.append(frame[::-1,:,:])
        obs = env.get_obs()
        a = pi.get_action(obs)[0]
        next_o, r, done, info = env.step(a)

env.close()
skvideo.io.vwrite('../videos/exo_arm.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
