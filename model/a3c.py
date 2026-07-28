import functools
import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.parameter import Parameter

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, dropout_flag=False, dropout=0.5, 
        output_activation=nn.Identity)->nn.Sequential:
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if dropout_flag:
            layers += [nn.Linear(sizes[j], sizes[j+1]), 
                       act(), nn.Dropout(dropout)]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class SimpleGCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(SimpleGCN, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    # adj_adjust is D^(-0.5)*(adj+I)*D^(0.5)
    def forward(self, h_0, adj_adjust):
        support = torch.matmul(h_0.float(), self.weight)
        output = torch.matmul(adj_adjust.float(), support)
        return output

"""
batch: return (batch_size, one-hot vector encoding for the graph)
one sample: return one-hot vector encoding for the graph
"""
class GCN(nn.Module):
    def __init__(self, feature_num, node_num, n_hidden, num_layer):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.feature_num = feature_num

        self.gcn_list = []
        for i in range(num_layer):
            if i == 0:
                self.gcn_list.append(SimpleGCN(feature_num, n_hidden))
            elif i == num_layer-1:
                self.gcn_list.append(SimpleGCN(n_hidden, feature_num))
            else:
                self.gcn_list.append(SimpleGCN(n_hidden, n_hidden))
        self.gcn_list = nn.ModuleList(self.gcn_list)

    # node_num: n
    # state_node: batch_size*n*feature_num
    # state_adj: batch_size*n*n
    # obs: batch_size*n*(feature_num+n)
    def forward(self, obs):
        # reconstruct state_node and state_adj from flatten_obs
        if (len(obs.size())==3):
            # batch
            adj_adjust, h_0 = torch.split(obs,
                [self.node_num, self.feature_num],dim=2)
        else:
            adj_adjust, h_0 = torch.split(obs,
                [self.node_num, self.feature_num],dim=1)

        for gcn in self.gcn_list:
            h_0 = F.relu(gcn(h_0, adj_adjust))

        if (len(h_0.size())==3):
            # batch
            bn_emb = torch.flatten(h_0,1)
        else:
            bn_emb = torch.flatten(h_0)
        return bn_emb


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class GCNCategoricalActor(Actor):

    def __init__(self, feature_num, node_num, gcn, 
                    hidden_sizes, act_num, activation):
        super().__init__()
        self.GCN = gcn
        self.logits_net = mlp([feature_num*node_num] + 
                              list(hidden_sizes) + [act_num], activation)

    # logits is the log probability, log_p = ln(p)
    def _distribution(self, obs):
        obs_emb = self.GCN(obs)
        logits = self.logits_net(obs_emb)
        return Categorical(logits=logits)

    def _get_logits(self, obs):
        obs_emb = self.GCN(obs)
        logits = self.logits_net(obs_emb)
        return logits

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GCNCritic(nn.Module):

    def __init__(self, feature_num, node_num, gcn, hidden_sizes, activation):
        super().__init__()
        self.GCN = gcn
        self.v_net = mlp([feature_num*node_num] + 
                            list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Squeeze is critical to ensure v has right shape.
        return torch.squeeze(self.v_net(self.GCN(obs)), -1) 


class GCNActorCritic(nn.Module):
   

    def __init__(self, observation_space, action_space, 
                 graph_encoder_hidden=256, num_gnn_layer=2, 
                 hidden_sizes=(64,64), activation=nn.ReLU, device='cpu'):
        super().__init__()
        self.device = device

        self.obs = []
        self.acts = []
        self.rews = []

        node_num = observation_space.shape[0]
        feature_num = observation_space.shape[1] - node_num
        
        act_num = action_space.n
        self.GCN = GCN(feature_num, node_num, 
                       graph_encoder_hidden, num_gnn_layer)
        self.pi = GCNCategoricalActor(feature_num, node_num, self.GCN, 
                                      hidden_sizes, act_num, activation)

        # build value function
        self.v = GCNCritic(feature_num, node_num, self.GCN, 
                           hidden_sizes, activation)
        params_num = sum(functools.reduce( lambda a, b: a*b, x.size()) 
                         for x in self.parameters())
        print("# of trainable params:{}".format(params_num))

    def step(self, obs, mask):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            
            pi_logits = self.pi._get_logits(obs)
            pi_logits_delta = torch.zeros(mask.size()).to(mask.device)
            pi_logits_delta[mask == 0] = float("-Inf")
            pi_logits += pi_logits_delta
            pi_mask = Categorical(logits=pi_logits)
            
            a = pi_mask.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)

            v = self.v(obs)
        
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
        

    def remember(self, obs, act, rew,):
        self.obs.append(obs)
        self.acts.append(float(act))
        self.rews.append(rew)

        
    def clear_memory(self):
        self.obs = []
        self.acts = []
        self.rews = []
    

    def forward(self, obs):
        gcnOut = self.GCN(obs)
        pi, _ = self.pi(gcnOut)
        v = self.v(gcnOut)
        
        return pi, v
    
    def choose_action(self, obs):
        gcnOut = self.GCN(obs)
        pi = self.pi._distribution(gcnOut)
        a = pi.sample()
        return a.item()
    
    def compute_loss_pi(self,data, clip_ratio):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def calc_R(self, done:bool):
        obs_array = np.array(self.obs, dtype=np.float32)
        states = torch.from_numpy(obs_array).to(self.device)
        # states = torch.tensor(self.obs, dtype=torch.float).to(self.device)
        v = self.v(states)
        returns = []   
        R = v[-1]*(1-int(done)) 

        for r in self.rews[::-1]:
            R = r + 0.95 * R
            returns.append(R)
        
        returns.reverse()
        return returns

    def calculate_advantages(rewards, values, discount_factor, trace_decay):
        advantages = []
        advantage = 0
        next_value = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * discount_factor - v
            advantage = td_error + advantage * discount_factor * trace_decay
            next_value = v
            advantages.insert(0, advantage)
            
        advantages = torch.tensor(advantages)

        return advantages
    
    def calculate_loss(self, done, discount_factor=0.95, trace_decay=0.97):

        # states = torch.tensor(self.obs, dtype=torch.float).to(self.device)
        # actions = torch.tensor(self.acts, dtype=torch.float).to(self.device)
        # rewards = torch.tensor(self.rews, dtype=torch.float).to(self.device)
        obs_array = np.array(self.obs, dtype=np.float32)
        states = torch.from_numpy(obs_array).to(self.device)
        action_array = np.array(self.acts, dtype=np.float32)
        actions = torch.from_numpy(action_array).to(self.device)
        rewards = np.array(self.rews, dtype=np.float32)
        rewards = torch.from_numpy(rewards).to(self.device)

        returns = self.calc_R(done)
        returns = torch.tensor(returns).to(self.device)
        values = self.v(states).squeeze()

        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * discount_factor - v
            advantage = td_error + advantage * discount_factor * trace_decay
            next_value = v
            advantages.insert(0, advantage)
            
        advantages = torch.tensor(advantages)
        advantages = advantages.detach()
        

        #normalize advantages
        mean = advantages.mean()
        std = advantages.std()
        normalized_advantages = (advantages - mean) / std
        
        returns = returns.detach()
        pi = self.pi._distribution(states)
        log_prob_actions = pi.log_prob(actions)
        
        actor_loss = - (normalized_advantages * log_prob_actions)

        value_loss = (returns - values)**2

        return actor_loss, value_loss
    
    # def calculate_loss(self, done):

    #     states = torch.tensor(self.obs, dtype=torch.float).to(self.device)
    #     actions = torch.tensor(self.acts, dtype=torch.float).to(self.device)
    #     rewards = torch.tensor(self.rews, dtype=torch.float).to(self.device)

    #     returns = self.calc_R(done)
    #     returns = torch.tensor(returns).to(self.device)
    #     v = self.v(states)
    #     pi = self.pi._distribution(states)

    #     values = v.squeeze()
    #     critic_loss = (returns-values)**2

    #     actor_loss = -pi.log_prob(actions)*(returns-values)

    #     return actor_loss, critic_loss


