import torch
import torch.nn.functional as F

import utils
from agent.networks.actor import Actor, Value
from agent.networks.sac_actor import SacActor
from agent.networks.critic import Critic


class Agent:
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            lr,
            hidden_dim,
            critic_target_tau,
            num_expl_steps,
            update_every_steps,
            stddev_schedule,
            stddev_clip,
            use_tb,
            alpha
    ):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        # Temperature parameter
        self.alpha = alpha

        # TODO: Define an actor network
        # self.actor = None
        self.actor = SacActor(obs_shape[0], action_shape[0], hidden_dim).to(self.device)

        self.critic1 = Critic(obs_shape[0], action_shape[0], hidden_dim).to(self.device)
        self.critic2 = Critic(obs_shape[0], action_shape[0], hidden_dim).to(self.device)

        self.value = Value(obs_shape[0], action_shape[0], hidden_dim).to(self.device)
        self.value_target = Value(obs_shape[0], action_shape[0], hidden_dim).to(self.device)
        # self.critic_target = Critic(obs_shape[0], action_shape[0], hidden_dim).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        # TODO: Define the optimizers for the actor and critic networks
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.value_target_opt = torch.optim.Adam(self.value_target.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def __repr__(self):
        return "sac"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic1.train(training)
        self.critic2.train(training)
        self.value.train(training)
        self.value_target.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)

        # stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        dist = self.actor(obs)

        if eval_mode:
            # TODO: Sample an action from the distribution in eval mode
            # action = None
            action = dist.mean
        else:
            if step < self.num_expl_steps:
                # TODO: Sample a random action between -1 and 1
                # pass
                action = torch.rand_like(dist.mean) * 2 - 1
            else:
                # TODO: Sample an action from the distribution
                # action = None
                action = dist.sample()

        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            # stddev = utils.schedule(self.stddev_schedule, step)
            # TODO: Compute the target Q value
            # Hint: Use next obs and next action to compute the target Q value
            # target_Q = None

            # next_action = self.actor(next_obs).sample()
            # target_Q = self.critic_target(next_obs, next_action)
            # y_i = reward + (discount * target_Q).detach()

            # V = self.value(obs)
            target_V = self.value_target(obs)



        # TODO: Compute the Q value from the critic network
        # Q = self.critic(obs, action)

        # TODO: Compute the critic loss
        # critic_loss = None
        # critic_loss = F.mse_loss(Q, y_i)
        # TODO: Optimize the critic network
        # self.critic_opt.zero_grad()
        # critic_loss.backward()
        # self.critic_opt.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.alpha * reward + self.gamma * target_V
        q1_old_policy = self.critic_1.forward(obs, action)
        q2_old_policy = self.critic_2.forward(obs, action)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_V.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        # and sample an action from the distribution
        # dist = None
        # action = None
        dist = self.actor(obs)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1_new_policy = self.critic_1.forward(obs, action)
        q2_new_policy = self.critic_2.forward(obs, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        # TODO: Get the Q value from the critic network
        # Q = None
        # Q = self.critic(obs, action)

        # TODO: Compute the actor loss and update
        actor_loss = log_prob - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        # TODO: Compute the value loss and update
        v = self.value(obs)
        v_target = critic_value - log_prob
        value_loss = 0.5 * F.mse_loss(v, v_target)
        self.value_opt.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # convert to float
        obs = obs.float()
        next_obs = next_obs.float()
        action, reward, discount = action.float(), reward.float(), discount.float()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.value, self.value_target, self.critic_target_tau
        )

        return metrics

    def save_snapshot(self):
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        return payload
