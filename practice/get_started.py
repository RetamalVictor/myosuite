import myosuite
import gym
import skvideo.io
import numpy as np
import os

env = gym.make('myoElbowPose1D6MRandom-v0')
print('List of cameras available',env.sim.model.camera_names)
env.reset()
frames = []
for _ in range(100):
    frame = env.sim.render(width=400, height=400,mode='offscreen', camera_name=None)
    frames.append(frame[::-1,:,:])
    env.step(env.action_space.sample()) # take a random action
env.close()

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/temp.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})