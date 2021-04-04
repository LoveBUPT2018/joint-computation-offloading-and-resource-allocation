# -*- coding: utf-8 -*-


import copy, json, argparse
import torch
from scenario import Scenario
from agent import Agent
from dotdic import DotDic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_agents(opt, sce, scenario, device):
	agents = []   # Vector of agents
	for i in range(opt.nagents):
		agents.append(Agent(opt, sce, scenario, index=i, device=device)) # Initialization, create a CNet for each agent
	return agents
    
def run_episodes(opt, sce, agents, scenario): 
    global_step = 0
    nepisode = 0
    action = torch.zeros(opt.nagents,dtype=int)
    reward = torch.zeros(opt.nagents)
    QoS = torch.zeros(opt.nagents)
    state_target = torch.ones(opt.nagents)  # The QoS requirement
    f= open("DDPG.csv","w+")
    f.write("This includes the running steps:\n")
    while nepisode < opt.nepisodes:
        state = torch.zeros(opt.nagents)  # Reset the state   
        next_state = torch.zeros(opt.nagents)  # Reset the next_state
        nstep = 0
        while nstep < opt.nsteps:            
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                # eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay) 
                # Exponential decay epsilon
            for i in range(opt.nagents):
                action[i] = agents[i].Select_Action(state, scenario, eps_threshold)  # Select action
            for i in range(opt.nagents):
                QoS[i], reward[i] = agents[i].Get_Reward(action, action[i], state, scenario)  # Obtain reward and next state
                next_state[i] = QoS[i]
            for i in range(opt.nagents):
                agents[i].Save_Transition(state, action[i], next_state, reward[i], scenario)  # Save the state transition
                agents[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    agents[i].Target_Update()
            state = copy.deepcopy(next_state)  # State transits 
            if torch.all(state.eq(state_target)):  # If QoS is satisified, break
                break
            nstep += 1  
        print('Episode Number:', nepisode, 'Training Step:', nstep)       
     #   print('Final State:', state)
        f.write("%i \n" % nstep)
        nepisode += 1
    f.close()
                
def run_trial(opt, sce):
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)  # Initialization 
    run_episodes(opt, sce, agents, scenario)    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    sce = DotDic(json.loads(open(args.config_path1, 'r').read()))
    opt = DotDic(json.loads(open(args.config_path2, 'r').read()))  # Load the configuration file as arguments
    for i in range(args.ntrials):
        trial_result_path = None
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)
















