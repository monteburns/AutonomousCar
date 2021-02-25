import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *

class TD3agent(nn.Module):
    def __init__(self, env, params, insize=23, device="cuda", discount=0.99,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        super().__init__()
        # Params
        self.num_states = insize
        self.num_actions = env.action_space.shape[0]
        self.gamma = params.gamma
        self.tau = params.tau
        self.device = device

        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.hidden_size = 256
        # Networks
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(
            self.device)
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions,
                             device=self.device).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions,
                                    device=self.device).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(params.buffersize)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.lrvalue)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.lrpolicy)

        self.total_it = 0

    def get_action(self, state):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]

        return action

    def update(self, batch_size):
        self.total_it += 1

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target.forward(next_states)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target.forward(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + dones * self.discount * target_Q

        # Critic loss
        current_Q1, current_Q2 = self.critic.forward(states, actions)
        critic_loss = self.critic_criterion(current_Q1, target_Q) + self.critic_criterion(current_Q2, target_Q)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(states, self.actor.forward(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)