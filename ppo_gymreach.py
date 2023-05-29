import random
import numpy as np
import torch
import argparse
import os
from distutils.util import strtobool
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

def make_env(gym_id, seed, capture_video, run_name):
    env = gym.make(gym_id,render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

    # Normalizer the environment   
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def _eval_agent(env, agent):
    total_success_rate = []
    for _ in range(50):
        per_success_rate = []
        observation, _ = env.reset()
        for _ in range(100):
            input_obs = _preproc_inputs(observation['observation'], observation['desired_goal'])

            next_obs = torch.Tensor(input_obs).to(device)  # store initial observation

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)  # roll out phase, no need gradient
                values[step] = value.flatten()
                action = action.cpu().numpy()
            action = action.flatten()

            observation_new, _, _, _, info = env.step(action)
            observation = observation_new
            per_success_rate.append(info['is_success'])
        
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    return local_success_rate

def _preproc_inputs(obs, g):
    # obs_norm = self.o_norm.normalize(obs)
    
    # obs_norm = self.normalize(obs)
    # g_norm = self.normalize(g)
    # g_norm = self.g_norm.normalize(g)
    # concatenate the stuffs
    inputs = np.concatenate([obs, g])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

    return inputs
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  
    # effective way to combat exploding and vanishing gradients, good for ppo # https://datascience.stackexchange.com/questions/64899/why-is-orthogonal-weights-initialization-so-important-for-ppo
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, input_shape):
        super(Agent, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(input_shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )

        # generate mean and std to have probability distribution 
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(input_shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01)  
        # 0.01 std ensure layers params have similar scalar values, results in the probability of taking each action will be similar
        )
        # std is state - independent
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="FetchReachDense-v3",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument('--wandb-project-name', type=str, default='cleanRL',
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="whether to capture videos of the agents performances")
    parser.add_argument('--save-dir', type=str, default='save_dir',
        help="save directory")
    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=1,
        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=2048,
        help='the total number of steps to run in each batch')
    parser.add_argument('--mini-steps', type=int, default=200,
        help='the number of steps to run in each trajectory')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="toggle learning rate annealing for policy and value network")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for GAE')
    parser.add_argument('--num-minibatches', type=int, default=32,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=10,
        help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles whether or not to use a clipped loss for the value function, as per the paper')
    parser.add_argument('--ent-coef', type=float, default=0.0,
        help='coefficient of the entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    args = parser.parse_args()
    args.batch_size = int(args.num_envs*args.num_steps)
    args.minibatch_size = int(args.batch_size//args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}_{args.exp_name}_toy_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,  # sync tensorboard metrics
            config=vars(args),
            name=run_name,
            monitor_gym=True,   # upload video in gym environment
            save_code=True,     # save a copy of code
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create multiple sub-env
    envs = make_env(args.gym_id, args.seed, args.capture_video, run_name)

    assert isinstance(envs.action_space, gym.spaces.Box), "only continous action space is supported"
    
    obs = envs.observation_space
    input_shape = obs["observation"].shape[0] + obs["desired_goal"].shape[0]

    print("envs.observation_space.shape", input_shape)
    print("envs.action_space.n", envs.action_space)

    agent = Agent(envs, input_shape).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) # default implement have eps=1e-5. default in pytorch eps=1e-8

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (input_shape,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start the game
    global_step = 0
    start_time = time.time()

    # observation, _ = envs.reset()
    # input_obs = _preproc_inputs(observation['observation'], observation['desired_goal'])

    # next_obs = torch.Tensor(input_obs).to(device)  # store initial observation
    next_done = torch.zeros(args.num_envs).to(device)  # initial termination condition
    num_updates = args.total_timesteps // args.batch_size  # number of update times

    # print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    # print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))
    print(f"[INFO] epoch = {num_updates} ")
    for update in range(1, num_updates+1):
        if args.anneal_lr:
            frac = 1.0 - (update-1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow  # update learning rate
        
        for step in range(0, args.num_steps):
            if step % args.mini_steps == 0:
                observation, _ = envs.reset()
                input_obs = _preproc_inputs(observation['observation'], observation['desired_goal'])

                next_obs = torch.Tensor(input_obs).to(device)  # store initial observation
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)  # roll out phase, no need gradient
                values[step] = value.flatten()
            
            action = action.flatten()
            values = values.flatten()

            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            obs_new = next_obs['observation']
            inputs_obs_new = _preproc_inputs(obs_new, next_obs["desired_goal"])


            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(inputs_obs_new).to(device), torch.Tensor([done]).to(device)

            if 'episode' in info.keys():
                writer.add_scalar("charts/episodic_return", info['episode']["r"], global_step)

            # PPO bootstrap value if environments are not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae: # General agvantage estimation
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                
                advantages = advantages.flatten()
                returns = advantages + values
            else: # common way: return - values
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (input_shape,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size) # get all index of a batch
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # debug variables
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()      # old formula
                    approx_kl = ((ratio-1) - logratio).mean() # new suggestion
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # fraction trigger clip ratio

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv: # perform advantage normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean())/ (mb_advantages.std() + 1e-8) # 1e-8 to avoid devision by 0
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)  # ratio clip
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # max of negatives, contrast with min of positive in paper

                # Value loss
                newvalue = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2 # MSE
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped-b_returns[mb_inds] ** 2) # MSE
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5*v_loss_max.mean()
                else:
                    v_loss = 0.5*((newvalue - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                # minimize policy loss, value loss, but maximize entropy loss => encourage agent to explore more
                loss = pg_loss -args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # perform global gradient clipping to prevent them from growing too large
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) 
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # calculate explained variance, tells you the value funtion is a good indicator of the return
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y==0 else 1 - np.var(y_true-y_pred)/var_y
        
        save_path = os.path.join(args.save_dir, run_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if update % 20 == 0:
            print("[INFO] Evaluate..")
            success_rate = _eval_agent(envs, agent)
            print(f"Total success rate is {success_rate}")
            torch.save(agent, os.path.join(save_path, 'model_{}.pt'.format(success_rate)))
            
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()