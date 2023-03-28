#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.MinMaxDDP import MinMaxDDP
from controllers.DDP_const import DDP_const
from controllers.WDDP_const import WDDP_const
from controllers.MinMaxDDP_const import MinMaxDDP_const

import gym
import DDP_Systems

import os
import pickle
from plot import summarize


def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()
    

def main(dist, baseline, sys, num_sim, num_samples, plot_results, render):
        
    path = "./results/{}/{}/".format(sys, dist)
    if not os.path.exists(path):
        os.makedirs(path)

    
    system = gym.make(args.system, dist=dist, N_samples=num_samples, num_sim=num_sim, path_=path, render=render)

    output_wddp_list = []
    
    
    #Initialize DDP controllers
    if baseline==1:
        ddp_baseline = DDP_const(system, path)
        ddp_baseline_name = 'box-DDP'
        output_baseline_list = []
        output_baseline_backward_list = []
    elif baseline==2:
        ddp_baseline = [MinMaxDDP(system, path), DDP_const(system, path)]
        ddp_baseline_name = ['GT-DDP', 'box-DDP']
        output_baseline_list = [[],[]]
        output_baseline_backward_list = [[], []]
    else:
        ddp_baseline_name = 'None'
        output_baseline_list = [[]]
        output_baseline_backward_list = [[]]

    
    wddp = WDDP_const(system, path)
    
    print('--------------------------')


    for i in range(num_sim):
           print('WDDP {}:'.format(i))
           output_wddp_backward = wddp.run(i)
           output_wddp = wddp.simulate(render, output_wddp_backward, i)
           output_wddp_list.append(output_wddp)
           
           print('      cost:', output_wddp['cost'][0,0], 'time:', output_wddp['comp_time'])

           
           if baseline==1:
               print('{} {}:'.format(ddp_baseline_name, i))
               output_baseline_backward = ddp_baseline.run(i)
               output_baseline = ddp_baseline.simulate(render, output_baseline_backward, i)
               output_baseline_list.append(output_baseline)
               
               print('      cost:', output_baseline['cost'][0,0], 'time:', output_baseline['comp_time'])
               print('--------------------------')
               
           elif baseline==2:
                for baseline_indx, ddp_baseline_ in enumerate(ddp_baseline):
                    output_baseline_backward = ddp_baseline_.run(i)
                    output_baseline_backward_list[baseline_indx].append(output_baseline_backward)
                    
                    print('{} {}:'.format(ddp_baseline_name[baseline_indx], i))
                    output_baseline = ddp_baseline_.simulate(render, output_baseline_backward_list[baseline_indx][0], i)
                    output_baseline_list[baseline_indx].append(output_baseline)
                        
                    print('      cost:', output_baseline['cost'][0,0], 'time:', output_baseline['comp_time'])
                    print('--------------------------')

    #Summarize and plot the results
    print('\n-------Summary-------')
    summarize(output_baseline_list, output_wddp_list, dist, path, num_sim, ddp_baseline_name, plot_results, args.render)
    
    save_data(path + 'wddp.pkl', output_wddp_list)
    save_data(path + 'baseline.pkl', output_baseline_list)
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--num_sim', required=False, default=1, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=10, type=int) #number of disturbance samples
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--system', required=False, default="Car-v0", type=str) #simulation type (single or multiple)
    parser.add_argument('--render', required=False, action="store_true") #render results
    parser.add_argument('--baseline', required=False, default=0, type=int) #simulation type (single or multiple)

    args = parser.parse_args()

    main(args.dist, args.baseline, args.system, args.num_sim, args.num_samples, args.plot, args.render)
