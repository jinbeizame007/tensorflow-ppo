import numpy as np
import tensorflow as tf
import argparse
import gym

from models import mlp_actor_critic
from replay_memory import ReplayMemory
from utils import proc_id, num_procs, mpi_fork, mpi_avg, count_vars, sync_all_params, MpiAdamOptimizer

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--clip_ratio', type=float, default=0.2)
parser.add_argument('--actor_lr', type=float, default=3e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--train_pi_iters', type=int, default=80)
parser.add_argument('--train_v_iters', type=int, default=80)
parser.add_argument('--lam', type=float, default=0.97)
parser.add_argument('--max_ep_len', type=int, default=1000)
parser.add_argument('--target_kl', type=float, default=0.01)
parser.add_argument('--save_freq', type=int, default=10)
args = parser.parse_args()

mpi_fork(args.cpu)

seed = 10000 * proc_id()
tf.set_random_seed(seed)
np.random.seed(seed)

env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Inputs to computation graph
x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
a_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
adv_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
ret_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

# Main outputs from computation graph
pi, logp, logp_pi, v = mlp_actor_critic(x_ph, a_ph)

# Need all placeholders in *this* order later (to zip with data from buffer)
all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

# Every step, get: action, value, and logprob
get_action_ops = [pi, v, logp_pi]

# Experience buffer
local_steps_per_epoch = int(args.steps / num_procs())
memory = ReplayMemory(obs_dim, act_dim, local_steps_per_epoch, args.gamma, args.lam)

# Count variables
var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])

# Objective functions
ratio = tf.exp(logp - logp_old_ph) # pi(a|s) / pi_old(a|s)
min_adv = tf.where(adv_ph>0, (1+args.clip_ratio)*adv_ph, (1-args.clip_ratio)*adv_ph)
actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
critic_loss = tf.reduce_mean((ret_ph - v)**2)

# Info (useful to watch during learning)
approx_kl = tf.reduce_mean(logp_old_ph - logp) # a sample estimate for KL-divergence, easy to compute
approx_ent = tf.reduce_mean(-logp) # a sample estimate for entropy, also easy to compute
clipped = tf.logical_or(ratio > (1+args.clip_ratio), ratio < (1-args.clip_ratio))
clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

# Optimizers
train_actor = MpiAdamOptimizer(learning_rate=args.actor_lr).minimize(actor_loss)
train_critic = MpiAdamOptimizer(learning_rate=args.critic_lr).minimize(critic_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Sync params across processes
sess.run(sync_all_params())

def update():
    inputs = {k:v for k,v in zip(all_phs, memory.get())}
    pi_l_old, v_l_old, ent = sess.run([actor_loss, critic_loss, approx_ent], feed_dict=inputs)

    # Training
    for i in range(args.train_pi_iters):
        _, kl = sess.run([train_actor, approx_kl], feed_dict=inputs)
        kl = mpi_avg(kl)
        if kl > 1.5 * args.target_kl:
            break
    for _ in range(args.train_v_iters):
        sess.run(train_critic, feed_dict=inputs)
    pi_l_new, v_l_new, kl, cf = sess.run([actor_loss, critic_loss, approx_kl, clipfrac], feed_dict=inputs)

obs, reward, done, episode_return, episode_len = env.reset(), 0, False, 0, 0

# Main loop: collect experience in env and update/log each epoch
for epoch in range(args.epochs):
    for t in range(local_steps_per_epoch):
        action, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: obs.reshape(1,-1)})

        # save
        memory.store(obs, action, reward, v_t, logp_t)

        obs, reward, done, _ = env.step(action[0])
        episode_return += reward
        episode_len += 1

        terminal = done or (episode_len == args.max_ep_len)
        if terminal or (t==local_steps_per_epoch-1):
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = reward if done else sess.run(v, feed_dict={x_ph: obs.reshape(1,-1)})
            memory.finish_path(last_val)
            print('epoch:', epoch, 'epi_ret:', episode_return)
            obs, reward, done, episode_return, episode_len = env.reset(), 0, False, 0, 0
            break

    # Perform PPO update!
    update()



