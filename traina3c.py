import json
import torch
import numpy as np
import torch.multiprocessing as mp

from model.env import CloudEnv
from model.a3c import GCNActorCritic
from model.synthetic import set_slo
import model.core as core

import os
import sys
import datetime
import argparse
import sys

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor([0])  # Ensure state['step'] is a singleton tensor
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Agent(mp.Process):
    def __init__(self, experiment_name, cloudConf, input_dim, n_actions, global_ac, gamma, pi_optim, vf_optim, name, global_ep_index, epochs):
        super(Agent, self).__init__()
        #creating a new actor critic model for each worker
        self.local_actor_critic = GCNActorCritic(input_dim, n_actions)
        self.global_ac = global_ac
        
        self.name = 'w%02i' % name

        self.episode_idx = global_ep_index
        port = 8000 + name

        log_dir = cloudConf[0]+"/"+self.name
        
        #each worker class will be making their own environment
        self.env = CloudEnv(log_dir, cloudConf[1], cloudConf[2], cloudConf[3], cloudConf[4], cloudConf[5], cloudConf[6], cloudConf[7], port=port)
        self.pi_optim = pi_optim
        self.vf_optim = vf_optim

        self.epochs = epochs
        self.steps_in_epoch = cloudConf[1]

        self.save_every = 10;
        self.exp_name = experiment_name
    
    def run(self):
        self.utils = {}
        self.rewards = {}
        self.breach = {}
        self.local_actor_critic.load_state_dict(
                            self.global_ac.state_dict())
        
        if self.name == 'w00':
            #check the folder where we are storing the model and if the dir does not exist, make one
            if not os.path.exists('a3c/'+self.exp_name+'/model'):
                os.makedirs('a3c/'+self.exp_name+'/model')
        
        while self.episode_idx.value < self.epochs:
            done = False
            o,m = self.env.reset()
            t_step = 1
            score = []
            self.local_actor_critic.clear_memory()
            last_done = 0
            utils_collection = []
            breach_collection = []
            util_score = 0
            reward_score = 0
            # while not done:
            while t_step <= self.steps_in_epoch:    
                #passing in the observation to make the action choice
                action, _, _ = self.local_actor_critic.step(torch.as_tensor(o), torch.as_tensor(m))
                obs, mask, reward, done, _, end_util, breach = self.env.step(action)
                utils_collection.append(end_util)
                score.append(reward)
                breach_collection.append(breach)
                self.local_actor_critic.remember(obs, action, reward)
                # done = False
                
                if t_step % self.steps_in_epoch == 0 or done:
                    if done and t_step - last_done < 2:
                        self.local_actor_critic.clear_memory()
                        t_step += 1
                        continue
                    critic_loss, actor_loss = self.local_actor_critic.calculate_loss(done)
                    
                    #calling the optimizer to update the weights
                    self.pi_optim.zero_grad()
                    self.vf_optim.zero_grad()
                    
                    #doing backward on the loss
                    critic_loss = critic_loss.mean()
                    actor_loss = actor_loss.mean()
                    
                    critic_loss.backward(retain_graph=True)
                    actor_loss.backward(retain_graph=True)

                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_ac.parameters()):
                        global_param._grad = local_param.grad
                    
                    self.vf_optim.step()
                    self.pi_optim.step()
                    
                    self.local_actor_critic.load_state_dict(
                            self.global_ac.state_dict())
                    
                    self.local_actor_critic.clear_memory()
                
                if t_step == self.steps_in_epoch or done:
                    if done:
                        last_done = t_step
                    # if t_step == self.steps_in_epoch:
                    #     #write the epoch number and avg score to the source file
                    #     with open(self.source_log, 'a') as f:
                    #         f.write(f"{self.episode_idx.value},{avg_score[-1]}\n")
                        
                if t_step >= self.steps_in_epoch:
                    self.utils[self.episode_idx.value] = utils_collection
                    util_score = np.mean(utils_collection)
                    utils_collection = []
                    self.rewards[self.episode_idx.value] = score
                    reward_score = np.mean(score)
                    score = []
                    self.breach[self.episode_idx.value] = breach_collection

                t_step += 1
                o = obs
                m = mask

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                # if self.episode_idx.value % 10 == 0:
                #save if it is the last episode or multiples of save_every
                if self.episode_idx.value == self.epochs:
                    torch.save(self.global_ac, 'a3c/'+self.exp_name+'/model/model'+str(self.episode_idx.value)+'.pt')
            print(self.name, 'Ep:', self.episode_idx.value, '| Loss: ', critic_loss.item(), actor_loss.item(), ' | Util: ', util_score, ' | Reward: ', reward_score ,flush=True)
        

        with open('a3c/'+self.exp_name+'/utils'+self.name+'.json', 'w') as f:
            json.dump(self.utils, f)
        with open('a3c/'+self.exp_name+'/rewards'+self.name+'.json', 'w') as f:
            json.dump(self.rewards, f)
        with open('a3c/'+self.exp_name+'/breach'+self.name+'.json', 'w') as f:
            json.dump(self.breach, f)


if __name__ == '__main__':
    hp_file = "model/configs/hyperparams.json"
    hp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), hp_file)
    hyperparams = json.load(open(hp_file, "r"))


    parser = argparse.ArgumentParser()
    parser.add_argument('-n',"--exp_name", help="exp name", type=str)
    parser.add_argument('-e',"--epochs", help="epochs", type=int)
    parser.add_argument('-c',"--clip_ratio", help="clip ratio (epsilon)", type=float)
    parser.add_argument('-k', "--knob", help="knob", type=float)
    parser.add_argument('-ncp', "--ncomp", help="NFV Components", type=int)
    parser.add_argument('-ncf', "--nconf", help="VM Instances", type=int)
    parser.add_argument('-nw', "--nworkers", help="Number of workers", type=int)
    args = parser.parse_args()

    if args.exp_name:
        hyperparams["exp_name"] = args.exp_name
    if args.clip_ratio:
        hyperparams["clip_ratio"] = float(args.clip_ratio)
    if args.knob:
        hyperparams["knob"] = float(args.knob)
    if args.ncomp:
        hyperparams["ncomp"] = int(args.ncomp)
    if args.nconf:
        hyperparams["nconf"] = int(args.nconf)
    if args.epochs:
        hyperparams["epochs"] = int(args.epochs)
    if args.nworkers:
        hyperparams["nworkers"] = int(args.nworkers)

    # slo = int(np.exp(np.random.randint(240,840)/100))
    # freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
    slo = 250
    freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
    # dict of knob values from the name of the experiment
    knob = hyperparams["knob"] # For over, under and near provisioning
    print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
    for i in range(hyperparams["nworkers"]):
        set_slo(slo, freq, knob, port=8000+i)

    budget = hyperparams["budget"]
    overrun_lim = hyperparams["budget_relax"]
    mode = hyperparams["mode"]
    threads = hyperparams["threads"]
    nconf = hyperparams["nconf"]
    ncomp = hyperparams["ncomp"]
    algo = hyperparams["algo"]
    steps_per_epoch = hyperparams["steps_per_epoch"]
    pi_lr = hyperparams["pi_lr"]
    vf_lr = hyperparams["vf_lr"]


    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


    path = "a3c/"+ hyperparams["exp_name"]
    #pass the following to the cloud environment st each agent will have the same thing
    #we need to have different environments for each agent but with same configuration
    #append the log directory with the eorker's identity inside Agent class
    cloudConf = [path, steps_per_epoch, budget, slo, overrun_lim, mode, nconf, ncomp]
    epochs = hyperparams["epochs"]

    lr = 1e-4

    #making a dummy env so that we can use its dimensions for the global ac network
    env = CloudEnv(cloudConf[0], cloudConf[1], cloudConf[2], cloudConf[3], cloudConf[4], cloudConf[5], cloudConf[6], cloudConf[7])
    
    input_dims = env.observation_space
    n_actions = env.action_space

    
    global_actor_critic = GCNActorCritic(env.observation_space, env.action_space)
    #the global actor critic needed the env to setup the nn only
    env.close()
    
    #share the global network's memory
    global_actor_critic.share_memory()
    
    #have to figure out this optimizer thing
    #optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
    #                    betas=(0.92, 0.999))
    pi_optim = SharedAdam(global_actor_critic.pi.parameters(), lr=pi_lr)
    v_optim = SharedAdam(global_actor_critic.v.parameters(), lr=vf_lr)
    
    global_ep = mp.Value('i', 0)
    sys.stdout.flush()

    workers = [Agent(experiment_name = hyperparams["exp_name"],
                    cloudConf=cloudConf,
                    global_ac=global_actor_critic,
                    pi_optim=pi_optim,
                    vf_optim=v_optim,
                    input_dim=input_dims,
                    n_actions=n_actions,
                    gamma=0.99,
                    name=i,
                    global_ep_index=global_ep,
                    # epochs = epochs) for i in range(mp.cpu_count())]
                    epochs = epochs) for i in range(hyperparams["nworkers"])]
    start_time = datetime.datetime.now()
    [w.start() for w in workers]
    [w.join() for w in workers]
    time_taken = datetime.datetime.now() - start_time
    #save the time taken in a txt file
    with open('a3c/'+hyperparams["exp_name"]+'/time.txt', 'w') as f:
        f.write(f"Time taken: {time_taken}")
    

    print("-----------------------------------")
    print(hyperparams["exp_name"]+" done in:", time_taken, "seconds")
    print("-----------------------------------")