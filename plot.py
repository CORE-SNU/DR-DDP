#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import argparse
import pickle
import os

def summarize(out_baseline_list, out_drddp_list, dist, path, num, ddp_baseline_name, plot_results=True, render=False):
    x_list, J_list, u_list, Vf_list, J_wc_list, ref_list, obs_list, r_obs_list, col_list = [], [], [], [], [], [], [], [], []
    time_list = []

    if out_baseline_list[0]==[]:
        Baseline = 0
    else:
        if len(out_baseline_list) == 1:
            Baseline = 1
            x_minmax_ddp_list, J_minmax_ddp_list, u_minmax_ddp_list, Vf_minmax_ddp_list, J_wc_minmax_ddp_list, ref_minmax_ddp_list, col_minmax_ddp_list = [], [], [], [], [], [], []
            time_minmax_ddp_list= []

        else:
            Baseline = 2
            x_minmax_ddp_list, J_minmax_ddp_list, u_minmax_ddp_list, Vf_minmax_ddp_list, J_wc_minmax_ddp_list, ref_minmax_ddp_list, col_minmax_ddp_list = [], [], [], [], [], [], []
            x_ddp_list, J_ddp_list, u_ddp_list, Vf_ddp_list, J_wc_ddp_list, ref_ddp_list, col_ddp_list = [], [], [], [], [], [], []
            time_minmax_ddp_list= []
            time_ddp_list= []

        
    for out in out_drddp_list:
         x_list.append(out['state_traj'])
         J_list.append(out['cost'])
         u_list.append(out['control_traj'])
         time_list.append(out['comp_time'])
         Vf_list.append(out['vf'])
         J_wc_list.append(out['J_wc'])
         ref_list.append(out['ref'])
    try:
        obs_list = out['obs']
        r_obs_list = out['r_obs']
        obs_list.append(obs_list[-1])
        r_obs_list.append(r_obs_list[-1])
        for out in out_drddp_list:
            col_list.append(out['col'])
    except:
        pass
         
    x_mean, J_mean, u_mean, Vf_mean, J_wc_mean, ref_mean = np.mean(x_list, axis=0), np.mean(J_list, axis=0), np.mean(u_list, axis=0), np.mean(Vf_list, axis=0), np.mean(J_wc_list, axis=0), np.mean(ref_list, axis=0)
    x_std, J_std, u_std, Vf_std, J_wc_std, ref_std = np.std(x_list, axis=0), np.std(J_list, axis=0), np.std(u_list, axis=0), np.std(Vf_list, axis=0), np.std(J_wc_list, axis=0), np.std(ref_list, axis=0)
    time_ar = np.array(time_list)
    J_ar = np.array(J_list)
    try:
        col_mean = np.mean(col_list)
        col_std = np.std(col_list)
    except:
        pass
        
       

    if Baseline == 1:
        for out in out_baseline_list:
             x_minmax_ddp_list.append(out['state_traj'])
             J_minmax_ddp_list.append(out['cost'])
             u_minmax_ddp_list.append(out['control_traj'])
             time_minmax_ddp_list.append(out['comp_time'])
             Vf_minmax_ddp_list.append(out['vf'])
             J_wc_minmax_ddp_list.append(out['J_wc'])
             ref_minmax_ddp_list.append(out['ref'])
             try:
                 col_minmax_ddp_list.append(out['col'])
             except:
                 pass

        x_minmax_ddp_mean, J_minmax_ddp_mean, u_minmax_ddp_mean, Vf_minmax_ddp_mean, J_wc_minmax_ddp_mean, ref_minmax_ddp_mean = np.mean(x_minmax_ddp_list, axis=0), np.mean(J_minmax_ddp_list, axis=0), np.mean(u_minmax_ddp_list, axis=0), np.mean(Vf_minmax_ddp_list, axis=0), np.mean(J_wc_minmax_ddp_list, axis=0), np.mean(ref_minmax_ddp_list, axis=0)
        x_minmax_ddp_std, J_minmax_ddp_std, u_minmax_ddp_std, Vf_minmax_ddp_std, J_wc_minmax_ddp_std, ref_minmax_ddp_std = np.std(x_minmax_ddp_list, axis=0), np.std(J_minmax_ddp_list, axis=0), np.std(u_minmax_ddp_list, axis=0), np.std(Vf_minmax_ddp_list, axis=0), np.std(J_wc_minmax_ddp_list, axis=0), np.std(ref_minmax_ddp_list, axis=0)
        time_minmax_ddp_ar = np.array(time_minmax_ddp_list)
        J_minmax_ddp_ar = np.array(J_minmax_ddp_list)
        try:
            col_minmax_ddp_mean = np.mean(col_minmax_ddp_list)
            col_minmax_ddp_std = np.std(col_minmax_ddp_list)
        except:
            pass
        
        
    elif Baseline == 2:
        for out in out_baseline_list[0]:
             x_minmax_ddp_list.append(out['state_traj'])
             J_minmax_ddp_list.append(out['cost'])
             u_minmax_ddp_list.append(out['control_traj'])
             time_minmax_ddp_list.append(out['comp_time'])
             Vf_minmax_ddp_list.append(out['vf'])
             J_wc_minmax_ddp_list.append(out['J_wc'])
             ref_minmax_ddp_list.append(out['ref'])
             try:
                 col_minmax_ddp_list.append(out['col'])
             except:
                 pass

        x_minmax_ddp_mean, J_minmax_ddp_mean, u_minmax_ddp_mean, Vf_minmax_ddp_mean, J_wc_minmax_ddp_mean, ref_minmax_ddp_mean = np.mean(x_minmax_ddp_list, axis=0), np.mean(J_minmax_ddp_list, axis=0), np.mean(u_minmax_ddp_list, axis=0), np.mean(Vf_minmax_ddp_list, axis=0), np.mean(J_wc_minmax_ddp_list, axis=0), np.mean(ref_minmax_ddp_list, axis=0)
        x_minmax_ddp_std, J_minmax_ddp_std, u_minmax_ddp_std, Vf_minmax_ddp_std, J_wc_minmax_ddp_std, ref_minmax_ddp_std = np.std(x_minmax_ddp_list, axis=0), np.std(J_minmax_ddp_list, axis=0), np.std(u_minmax_ddp_list, axis=0), np.std(Vf_minmax_ddp_list, axis=0), np.std(J_wc_minmax_ddp_list, axis=0), np.std(ref_minmax_ddp_list, axis=0)
        time_minmax_ddp_ar = np.array(time_minmax_ddp_list)
        J_minmax_ddp_ar = np.array(J_minmax_ddp_list)
        try:
            col_minmax_ddp_mean = np.mean(col_minmax_ddp_list)
            col_minmax_ddp_std = np.std(col_minmax_ddp_list)
        except:
            pass
        
        for out in out_baseline_list[1]:
             x_ddp_list.append(out['state_traj'])
             J_ddp_list.append(out['cost'])
             u_ddp_list.append(out['control_traj'])
             time_ddp_list.append(out['comp_time'])
             Vf_ddp_list.append(out['vf'])
             J_wc_ddp_list.append(out['J_wc'])
             ref_ddp_list.append(out['ref'])
             try:
                 col_ddp_list.append(out['col'])
             except:
                 pass
             
        x_ddp_mean, J_ddp_mean, u_ddp_mean, Vf_ddp_mean, J_wc_ddp_mean, ref_ddp_mean = np.mean(x_ddp_list, axis=0), np.mean(J_ddp_list, axis=0), np.mean(u_ddp_list, axis=0), np.mean(Vf_ddp_list, axis=0), np.mean(J_wc_ddp_list, axis=0), np.mean(ref_ddp_list, axis=0)
        x_ddp_std, J_ddp_std, u_ddp_std, Vf_ddp_std, J_wc_ddp_std, ref_ddp_std = np.std(x_ddp_list, axis=0), np.std(J_ddp_list, axis=0), np.std(u_ddp_list, axis=0), np.std(Vf_ddp_list, axis=0), np.std(J_wc_ddp_list, axis=0), np.std(ref_ddp_list, axis=0)
        time_ddp_ar = np.array(time_ddp_list)
        J_ddp_ar = np.array(J_ddp_list)
        
        try:
            col_ddp_mean = np.mean(col_ddp_list)
            col_ddp_std = np.std(col_ddp_list)
        except:
            pass
        
            
    if plot_results:
        
        nx = x_mean.shape[1]
        T = u_mean.shape[0]
        nu = u_mean.shape[1]

        fig = plt.figure(figsize=(6,4), dpi=300)
        

        t = np.arange(T+1)
        for i in range(nx):
            plt.plot(t, x_mean[:,i,0], 'tab:blue')
            plt.fill_between(t, x_mean[:,i,0] + 0.3*x_std[:,i,0],
                               x_mean[:,i,0] - 0.3*x_std[:,i,0], facecolor='tab:blue', alpha=0.3)
                
        plt.xlabel(r'$t$', fontsize=22)
        plt.ylabel(r'States', fontsize=22)
        plt.legend(fontsize=20)
        plt.grid()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([t[0], t[-1]])
        ax = fig.gca()
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='x', nbins=5)
        fig.set_size_inches(6, 4)
        plt.savefig(path +'all_states_DR_DDP_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()
            
        
        if Baseline==1:
            for i in range(nx):
                plt.plot(t, x_minmax_ddp_mean[:,i,0], 'tab:red')
                plt.fill_between(t, x_minmax_ddp_mean[:,i, 0] + 0.3*x_minmax_ddp_std[:,i,0],
                               x_minmax_ddp_mean[:,i,0] - 0.3*x_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'States', fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'all_states_{}_{}.pdf'.format(ddp_baseline_name, num), dpi=300, bbox_inches="tight")
            plt.clf()
                
        elif Baseline==2:
            for i in range(nx):
                plt.plot(t, x_minmax_ddp_mean[:,i,0], 'tab:red')
                plt.fill_between(t, x_minmax_ddp_mean[:,i, 0] + 0.3*x_minmax_ddp_std[:,i,0],
                               x_minmax_ddp_mean[:,i,0] - 0.3*x_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'States', fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'all_states_{}_{}.pdf'.format(ddp_baseline_name[0], num), dpi=300, bbox_inches="tight")
            plt.clf()
            
            for i in range(nx):
            
                plt.plot(t, x_ddp_mean[:,i,0], 'tab:green')
                plt.fill_between(t, x_ddp_mean[:,i, 0] + 0.3*x_ddp_std[:,i,0],
                               x_ddp_mean[:,i,0] - 0.3*x_ddp_std[:,i,0], facecolor='tab:green', alpha=0.3)
        
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'States'.format(i+1), fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'all_states_{}_{}.pdf'.format(ddp_baseline_name[1], num), dpi=300, bbox_inches="tight")
            plt.clf()
                
            
        t = np.arange(T+1)
        for i in range(nx):
            try:
                plt.plot(t, ref_mean[:,i,0], 'tab:purple', label='Reference')
                plt.fill_between(t, ref_mean[:,i,0] + 0.3*ref_std[:,i,0],
                                   ref_mean[:,i,0] - 0.3*ref_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            except:
                pass

            if Baseline==1:
                plt.plot(t, x_minmax_ddp_mean[:,i,0], 'tab:red', label=ddp_baseline_name)
                plt.fill_between(t, x_minmax_ddp_mean[:,i, 0] + 0.3*x_minmax_ddp_std[:,i,0],
                               x_minmax_ddp_mean[:,i,0] - 0.3*x_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)
                
            elif Baseline==2:
                plt.plot(t, x_minmax_ddp_mean[:,i,0], 'tab:red', label=ddp_baseline_name[0])
                plt.fill_between(t, x_minmax_ddp_mean[:,i, 0] + 0.3*x_minmax_ddp_std[:,i,0],
                               x_minmax_ddp_mean[:,i,0] - 0.3*x_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)
                
                plt.plot(t, x_ddp_mean[:,i,0], 'tab:green', label=ddp_baseline_name[1])
                plt.fill_between(t, x_ddp_mean[:,i, 0] + 0.3*x_ddp_std[:,i,0],
                               x_ddp_mean[:,i,0] - 0.3*x_ddp_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
            plt.plot(t, x_mean[:,i,0], 'tab:blue', label='DR-DDP')
            plt.fill_between(t, x_mean[:,i,0] + 0.3*x_std[:,i,0],
                               x_mean[:,i,0] - 0.3*x_std[:,i,0], facecolor='tab:blue', alpha=0.3)
                
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'$x_{{{}}}$'.format(i+1), fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'states_{}_{}.pdf'.format(i+1, num), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T)
        for i in range(nu):

            if Baseline==1:
                plt.plot(t, u_minmax_ddp_mean[:,i,0], 'tab:red', label=ddp_baseline_name)
                plt.fill_between(t, u_minmax_ddp_mean[:,i,0] + u_minmax_ddp_std[:,i,0],
                             u_minmax_ddp_mean[:,i,0] - u_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)

            elif Baseline==2:
                plt.plot(t, u_minmax_ddp_mean[:,i,0], 'tab:red', label=ddp_baseline_name[0])
                plt.fill_between(t, u_minmax_ddp_mean[:,i,0] + u_minmax_ddp_std[:,i,0],
                             u_minmax_ddp_mean[:,i,0] - u_minmax_ddp_std[:,i,0], facecolor='tab:red', alpha=0.3)
        
                plt.plot(t, u_ddp_mean[:,i,0], 'tab:green', label=ddp_baseline_name[1])
                plt.fill_between(t, u_ddp_mean[:,i,0] + u_ddp_std[:,i,0],
                             u_ddp_mean[:,i,0] - u_ddp_std[:,i,0], facecolor='tab:green', alpha=0.3)
        
            plt.plot(t, u_mean[:,i,0], 'tab:blue', label='DR-DDP')
            plt.fill_between(t, u_mean[:,i,0] + u_std[:,i,0],
                             u_mean[:,i,0] - u_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$u_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlim([t[0], t[-1]])

            plt.savefig(path +'controls_{}_{}.pdf'.format(i+1, num), dpi=300, bbox_inches="tight")
            plt.clf()
        
        
           
            
        len_ = len(Vf_list[0])
        if Baseline==1:
            len_minmax_ddp = len(Vf_minmax_ddp_list[0])
            max_len = max([len_,len_minmax_ddp])
            Vf_mean = np.concatenate([Vf_mean, [Vf_mean[-1]]*abs(max_len-len_)])
            Vf_std = np.concatenate([Vf_std, [Vf_std[-1]]*abs(max_len-len_)])
            Vf_minmax_ddp_mean = np.concatenate([Vf_minmax_ddp_mean, [Vf_minmax_ddp_mean[-1]]*abs(max_len-len_minmax_ddp)])
            Vf_minmax_ddp_std = np.concatenate([Vf_minmax_ddp_std, [Vf_minmax_ddp_std[-1]]*abs(max_len-len_minmax_ddp)])
            t = np.arange(max_len)
        elif Baseline==2:
            len_minmax_ddp = len(Vf_minmax_ddp_list[0])
            len_ddp = len(Vf_ddp_list[0])
            max_len = max([len_,len_minmax_ddp,len_ddp])
            Vf_mean = np.concatenate([Vf_mean, [Vf_mean[-1]]*abs(max_len-len_)])
            Vf_std = np.concatenate([Vf_std, [Vf_std[-1]]*abs(max_len-len_)])
            Vf_minmax_ddp_mean = np.concatenate([Vf_minmax_ddp_mean, [Vf_minmax_ddp_mean[-1]]*abs(max_len-len_minmax_ddp)])
            Vf_minmax_ddp_std = np.concatenate([Vf_minmax_ddp_std, [Vf_minmax_ddp_std[-1]]*abs(max_len-len_minmax_ddp)])
            Vf_ddp_mean = np.concatenate([Vf_ddp_mean, [Vf_ddp_mean[-1]]*abs(max_len-len_ddp)])
            Vf_ddp_std = np.concatenate([Vf_ddp_std, [Vf_ddp_std[-1]]*abs(max_len-len_ddp)])
            t = np.arange(max_len)
        else:
            t = np.arange(len_)
        if Baseline==1:
            plt.plot(t, Vf_minmax_ddp_mean, 'tab:red', label=ddp_baseline_name)
            plt.fill_between(t, Vf_minmax_ddp_mean + 0.3*Vf_minmax_ddp_std, Vf_minmax_ddp_mean - 0.3*Vf_minmax_ddp_std, facecolor='tab:red', alpha=0.3)
        elif Baseline==2:
            plt.plot(t, Vf_minmax_ddp_mean, 'tab:red', label=ddp_baseline_name[0])
            plt.fill_between(t, Vf_minmax_ddp_mean + 0.3*Vf_minmax_ddp_std, Vf_minmax_ddp_mean - 0.3*Vf_minmax_ddp_std, facecolor='tab:red', alpha=0.3)
    
            plt.plot(t, Vf_ddp_mean, 'tab:green', label=ddp_baseline_name[1])
            plt.fill_between(t, Vf_ddp_mean + 0.3*Vf_ddp_std, Vf_ddp_mean - 0.3*Vf_ddp_std, facecolor='tab:green', alpha=0.3)

        plt.plot(t, Vf_mean, 'tab:blue', label='DR-DDP')
        plt.fill_between(t, Vf_mean + 0.3*Vf_std, Vf_mean - 0.3*Vf_std, facecolor='tab:blue', alpha=0.3)
        plt.xlabel(r'$iter$', fontsize=16)
        plt.ylabel('Optimal Value', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'V_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()

        len_ = len(J_wc_list[0])
        if Baseline==1:
            len_minmax_ddp = len(J_wc_minmax_ddp_list[0])
            max_len = max([len_,len_minmax_ddp])
            J_wc_mean = np.concatenate([J_wc_mean, [J_wc_mean[-1]]*abs(max_len-len_)])
            J_wc_std = np.concatenate([J_wc_std, [J_wc_std[-1]]*abs(max_len-len_)])
            J_wc_minmax_ddp_mean = np.concatenate([J_wc_minmax_ddp_mean, [J_wc_minmax_ddp_mean[-1]]*abs(max_len-len_minmax_ddp)])
            J_wc_minmax_ddp_std = np.concatenate([J_wc_minmax_ddp_std, [J_wc_minmax_ddp_std[-1]]*abs(max_len-len_minmax_ddp)])
            t = np.arange(max_len)
        elif Baseline==2:
            len_minmax_ddp = len(J_wc_minmax_ddp_list[0])
            len_ddp = len(J_wc_ddp_list[0])
            max_len = max([len_,len_minmax_ddp,len_ddp])
            J_wc_mean = np.concatenate([J_wc_mean, [J_wc_mean[-1]]*abs(max_len-len_)])
            J_wc_std = np.concatenate([J_wc_std, [J_wc_std[-1]]*abs(max_len-len_)])
            J_wc_minmax_ddp_mean = np.concatenate([J_wc_minmax_ddp_mean, [J_wc_minmax_ddp_mean[-1]]*abs(max_len-len_minmax_ddp)])
            J_wc_minmax_ddp_std = np.concatenate([J_wc_minmax_ddp_std, [J_wc_minmax_ddp_std[-1]]*abs(max_len-len_minmax_ddp)])
            J_wc_ddp_mean = np.concatenate([J_wc_ddp_mean, [J_wc_ddp_mean[-1]]*abs(max_len-len_ddp)])
            J_wc_ddp_std = np.concatenate([J_wc_ddp_std, [J_wc_ddp_std[-1]]*abs(max_len-len_ddp)])
            t = np.arange(max_len)
        else:
            t = np.arange(len_)
            
        if Baseline==1:
            plt.plot(t, J_wc_minmax_ddp_mean, 'tab:red', label=ddp_baseline_name)
            plt.fill_between(t, J_wc_minmax_ddp_mean + 0.3*J_wc_minmax_ddp_std, J_wc_minmax_ddp_mean - 0.3*J_wc_minmax_ddp_std, facecolor='tab:red', alpha=0.3)
        elif Baseline==2:
            plt.plot(t, J_wc_minmax_ddp_mean, 'tab:red', label=ddp_baseline_name[0])
            plt.fill_between(t, J_wc_minmax_ddp_mean + 0.3*J_wc_minmax_ddp_std, J_wc_minmax_ddp_mean - 0.3*J_wc_minmax_ddp_std, facecolor='tab:red', alpha=0.3)
            plt.plot(t, J_wc_ddp_mean, 'tab:green', label=ddp_baseline_name[1])
            plt.fill_between(t, J_wc_ddp_mean + 0.3*J_wc_ddp_std, J_wc_ddp_mean - 0.3*J_wc_ddp_std, facecolor='tab:green', alpha=0.3)
    
        plt.plot(t, J_wc_mean, 'tab:blue', label='DR-DDP')
        plt.fill_between(t, J_wc_mean + 0.3*J_wc_std, J_wc_mean - 0.3*J_wc_std, facecolor='tab:blue', alpha=0.3)
        plt.xlabel(r'$iter$', fontsize=16)
        plt.ylabel('Worst-Case Cost', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_wc_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()
        
        
        ax = fig.gca()
        t = np.arange(T+1)
        
        if Baseline==1:
            max_bin = np.max([J_ar[:,0], J_minmax_ddp_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_minmax_ddp_ar[:,0]])
        elif Baseline==2:
            max_bin = np.max([J_ar[:,0], J_minmax_ddp_ar[:,0], J_ddp_ar[:,0]])
            min_bin = np.min([J_ar[:,0], J_minmax_ddp_ar[:,0], J_ddp_ar[:,0]])
        else:
            max_bin = np.max(J_ar[:,0])
            min_bin = np.min(J_ar[:,0])
            
        ax.hist(J_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='DR-DDP', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        
        if Baseline==1:
            ax.hist(J_minmax_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name, alpha=0.5, linewidth=0.5, edgecolor='tab:red')
        elif Baseline==2:
            ax.hist(J_minmax_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name[0], alpha=0.5, linewidth=0.5, edgecolor='tab:red')
            ax.hist(J_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:green', label=ddp_baseline_name[1], alpha=0.5, linewidth=0.5, edgecolor='tab:green')

        ax.axvline(J_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        
        if Baseline==1:
            ax.axvline(J_minmax_ddp_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
        elif Baseline==2:
            ax.axvline(J_minmax_ddp_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
            ax.axvline(J_ddp_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        if Baseline==1:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1, 0]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
        elif Baseline==2:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [2, 1, 0]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

        ax.grid()
        ax.set_axisbelow(True)
        plt.xlabel(r'Total Cost', fontsize=16)
        plt.ylabel(r'Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_hist_comp_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()


        ax = fig.gca()
        t = np.arange(T+1)
        
        max_bin = np.max(J_ar[:,0])
        min_bin = np.min(J_ar[:,0])
            
        ax.hist(J_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='DR-DDP', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        
        ax.axvline(J_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.grid()
        ax.set_axisbelow(True)
        plt.xlabel(r'Total Cost (DR-DDP)', fontsize=16)
        plt.ylabel(r'Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_hist_DDP_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
        plt.clf()

        if Baseline==1:
            ax = fig.gca()
            t = np.arange(T+1)
            
            max_bin = np.max(J_minmax_ddp_ar[:,0])
            min_bin = np.min(J_minmax_ddp_ar[:,0])
                
            
            ax.hist(J_minmax_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name, alpha=0.5, linewidth=0.5, edgecolor='tab:red')
            ax.axvline(J_minmax_ddp_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
            
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    
            ax.grid()
            ax.set_axisbelow(True)
            plt.xlabel(r'Total Cost ({})'.format(ddp_baseline_name), fontsize=16)
            plt.ylabel(r'Frequency', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(path +'J_hist_{}_{}.pdf'.format(ddp_baseline_name, num), dpi=300, bbox_inches="tight")
            plt.clf()
        
        elif Baseline==2:
            ax = fig.gca()
            t = np.arange(T+1)
            
            max_bin = np.max(J_minmax_ddp_ar[:,0])
            min_bin = np.min(J_minmax_ddp_ar[:,0])
                
            
            ax.hist(J_minmax_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name[0], alpha=0.5, linewidth=0.5, edgecolor='tab:red')
            ax.axvline(J_minmax_ddp_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
            
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    
            ax.grid()
            ax.set_axisbelow(True)
            plt.xlabel(r'Total Cost ({})'.format(ddp_baseline_name[0]), fontsize=16)
            plt.ylabel(r'Frequency', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(path +'J_hist_{}_{}.pdf'.format(ddp_baseline_name[0], num), dpi=300, bbox_inches="tight")
            plt.clf()
            
            ax = fig.gca()
            t = np.arange(T+1)
            
            max_bin = np.max(J_ddp_ar[:,0])
            min_bin = np.min(J_ddp_ar[:,0])
                
            
            ax.hist(J_ddp_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:green', label=ddp_baseline_name[1], alpha=0.5, linewidth=0.5, edgecolor='tab:green')
            ax.axvline(J_ddp_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)
            
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    
            ax.grid()
            ax.set_axisbelow(True)
            plt.xlabel(r'Total Cost ({})'.format(ddp_baseline_name[1]), fontsize=16)
            plt.ylabel(r'Frequency', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(path +'J_hist_{}_{}.pdf'.format(ddp_baseline_name[1], num), dpi=300, bbox_inches="tight")
            plt.clf()
            
            
        try:
            ax = fig.gca()
            t = np.arange(T+1)
            col_ar = np.array(col_list)
            if Baseline==1:
                col_minmax_ddp_ar = np.array(col_minmax_ddp_list)
            elif Baseline==2:
                col_minmax_ddp_ar = np.array(col_minmax_ddp_list)
                col_ddp_ar = np.array(col_ddp_list)
    
            if Baseline==1:
                max_bin = np.max([col_ar, col_minmax_ddp_ar])
                min_bin = np.min([col_ar, col_minmax_ddp_ar])
            elif Baseline==2:
                max_bin = np.max([col_ar, col_minmax_ddp_ar, col_ddp_ar])
                min_bin = np.min([col_ar, col_minmax_ddp_ar, col_ddp_ar])
            else:
                max_bin = np.max(col_ar)
                min_bin = np.min(col_ar)
                
            ax.hist(J_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='DR-DDP', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
            
            if Baseline==1:
                ax.hist(col_minmax_ddp_ar, bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name, alpha=0.5, linewidth=0.5, edgecolor='tab:red')
            elif Baseline==2:
                ax.hist(col_minmax_ddp_ar, bins=50, range=(min_bin,max_bin), color='tab:red', label=ddp_baseline_name[0], alpha=0.5, linewidth=0.5, edgecolor='tab:red')
                ax.hist(col_ddp_ar, bins=50, range=(min_bin,max_bin), color='tab:orange', label=ddp_baseline_name[1], alpha=0.5, linewidth=0.5, edgecolor='tab:green')
    
            ax.axvline(col_ar.mean(), color='navy', linestyle='dashed', linewidth=1.5)
            
            if Baseline==1:
                ax.axvline(col_minmax_ddp_ar.mean(), color='maroon', linestyle='dashed', linewidth=1.5)
            elif Baseline==2:
                ax.axvline(col_minmax_ddp_ar.mean(), color='maroon', linestyle='dashed', linewidth=1.5)
                ax.axvline(col_ddp_ar.mean(), color='orange', linestyle='dashed', linewidth=1.5)
    
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    
            if Baseline==1:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [1, 0]
                ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
            elif Baseline==2:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [2, 1, 0]
                ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
    
            ax.grid()
            ax.set_axisbelow(True)
            plt.xlabel(r'Number of Collisions', fontsize=16)
            plt.ylabel(r'Frequency', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(path +'Col_hist_comp_{}.pdf'.format(num), dpi=300, bbox_inches="tight")
            plt.clf()
                
            
            fig = plt.figure(figsize=(6,6), dpi=300)
            
            T = x_mean.shape[0]
            
            colors = ['rab:green', 'tab:green']
            # obs_face_color = [148/255, 103/255, 189/255, 0.05]
            # obs_edge_color = [148/255, 103/255, 189/255, 0.1]
            
            orange = list(cl.to_rgba('darkseagreen'))
            blue = 'indianred'
            green = 'royalblue'
            red = 'goldenrod'
            obs_face_color = orange.copy()
            obs_face_color[-1] = 0.05

            obs_edge_color = orange.copy()
            obs_edge_color[-1] = 0.1

            obs_patch_list = []
            obs_traj_plot_list = []
            

            
            def check_col(x, obs, r_obs):
                if ((x-obs).T @ (x - obs))**0.5 <= r_obs + 0.2:
                    return True
                else:
                    return False
            
            def update_graph(num):
                # for obs in list(obs_patch_list):
                #     obs.remove()
                #     obs_patch_list.remove(obs)
                if num >= 79:
                    print(num, 'out of', T)
                
                for obs_indx in range(len(obs_list[num])):
                    obs_patch = ax.add_patch(plt.Circle((x_mean[num,3+2*obs_indx], x_mean[num,3+2*obs_indx+1]),0.3, facecolor=obs_face_color, edgecolor=obs_edge_color))
                    obs_patch_list.append(obs_patch)
                    obs_traj_plot_list[obs_indx].set_data(x_mean[:num,3+2*obs_indx,0], x_mean[:num,3+2*obs_indx+1,0])
                        
                    
                # ref_plot.set_data(ref_mean[:num,0,0], ref_mean[:num,1,0])
                # ref_plot.set_3d_properties(ref_mean[:num,2,0])
                traj_plot.set_data(x_mean[:num,0,0], x_mean[:num,1,0])
                ref_plot_.set_data(ref_mean[num,0], ref_mean[num,1])
                traj_plot_.set_data(x_mean[num,0], x_mean[num,1])
                # for obs_indx in range(len(obs_list[num])):
                #     if check_col(x_mean[num,:2,0], x_mean[num,3+2*obs_indx:3+2*(obs_indx+1),0], 0.3):
                #         ax.plot(x_mean[num,0,0], x_mean[num,1,0], marker="*", color='blue')
                
                
                if Baseline==1:
                    traj_minmax_ddp_plot.set_data(x_minmax_ddp_mean[:num,0,0], x_minmax_ddp_mean[:num,1,0])
                    traj_minmax_ddp_plot_.set_data(x_minmax_ddp_mean[num,0], x_minmax_ddp_mean[num,1])
                    for obs_indx in range(len(obs_list[num])):
                        if check_col(x_minmax_ddp_mean[num,:2,0], x_minmax_ddp_mean[num,3+2*obs_indx:3+2*(obs_indx+1),0], 0.3):
                            ax.plot(x_minmax_ddp_mean[num,0,0], x_minmax_ddp_mean[num,1,0], marker="*", markerfacecolor=red, markeredgecolor='black', markeredgewidth=0.5)
                
                elif Baseline==2:
                    traj_minmax_ddp_plot.set_data(x_minmax_ddp_mean[:num,0,0], x_minmax_ddp_mean[:num,1,0])
                    traj_minmax_ddp_plot_.set_data(x_minmax_ddp_mean[num,0], x_minmax_ddp_mean[num,1])
                    for obs_indx in range(len(obs_list[num])):
                        if check_col(x_minmax_ddp_mean[num,:2,0], x_minmax_ddp_mean[num,3+2*obs_indx:3+2*(obs_indx+1),0], 0.3):
                            ax.plot(x_minmax_ddp_mean[num,0,0], x_minmax_ddp_mean[num,1,0], marker="*", markerfacecolor=red, markeredgecolor='black', markeredgewidth=0.5)
                
                    traj_ddp_plot.set_data(x_ddp_mean[:num,0,0], x_ddp_mean[:num,1,0])
                    traj_ddp_plot_.set_data(x_ddp_mean[num,0], x_ddp_mean[num,1])
                    for obs_indx in range(len(obs_list[num])):
                        if check_col(x_ddp_mean[num,:2,0], x_ddp_mean[num,3+2*obs_indx:3+2*(obs_indx+1),0], 0.3):
                            ax.plot(x_ddp_mean[num,0,0], x_ddp_mean[num,1,0], marker="*", markerfacecolor=green, markeredgecolor='black', markeredgewidth=0.5)
                
    
            fig, ax = plt.subplots()
    
            ax.add_patch(plt.Rectangle((-5, -5), 10, 10, color=[204/255, 204/255, 204/255]))
            ax.add_patch(plt.Rectangle((-5, -1), 10, 2, color=[1, 1, 1]))
            ax.add_patch(plt.Rectangle((-1, -5), 2, 10, color=[1, 1, 1]))


            ax.plot([-5, -1],[1, 1], linewidth=1, color='black')
            ax.plot([-5, -1],[-1, -1], linewidth=1, color='black')
            ax.plot([1, 5],[1, 1], linewidth=1, color='black')
            ax.plot([1, 5],[-1, -1], linewidth=1, color='black')
            ax.plot([-1, -1],[1, 5], linewidth=1, color='black')
            ax.plot([-1, -1],[-1, -5], linewidth=1, color='black')
            ax.plot([1, 1],[1, 5], linewidth=1, color='black')
            ax.plot([1, 1],[-1, -5], linewidth=1, color='black')
            
            ax.plot([-5, -1],[0, 0], linestyle='--', linewidth=1, color='black')
            ax.plot([1, 5],[0, 0], linestyle='--', linewidth=1, color='black')
            ax.plot([0, 0],[1, 5], linestyle='--', linewidth=1, color='black')
            ax.plot([0, 0],[-1, -5], linestyle='--', linewidth=1, color='black')

    
                
            ref_plot_ = ax.plot(ref_mean[:T,0,0], ref_mean[:T,1,0], marker='o', color='black')[0]
            ref_plot = ax.plot(ref_mean[:T,0,0], ref_mean[:T,1,0], '--', color='black')[0]

            
            if Baseline==1:
                traj_minmax_ddp_plot_ = ax.plot(x_minmax_ddp_mean[0,0,0], x_minmax_ddp_mean[0,1,0], marker='o', color=red)[0]
                traj_minmax_ddp_plot = ax.plot(x_minmax_ddp_mean[0,0,0], x_minmax_ddp_mean[0,1,0], color=red, label=ddp_baseline_name)[0]
            elif Baseline==2:
                traj_minmax_ddp_plot_ = ax.plot(x_minmax_ddp_mean[0,0,0], x_minmax_ddp_mean[0,1,0], marker='o', color=red)[0]
                traj_minmax_ddp_plot = ax.plot(x_minmax_ddp_mean[0,0,0], x_minmax_ddp_mean[0,1,0], color=red, label=ddp_baseline_name[0])[0]
                traj_ddp_plot_ = ax.plot(x_ddp_mean[0,0,0], x_ddp_mean[0,1,0], marker='o', color=green)[0]
                traj_ddp_plot = ax.plot(x_ddp_mean[0,0,0], x_ddp_mean[0,1,0],  color=green, label=ddp_baseline_name[1])[0]
                
            traj_plot_ = ax.plot(x_mean[0,0,0], x_mean[0,1,0], marker='o', color=blue)[0]
            traj_plot = ax.plot(x_mean[0,0,0], x_mean[0,1,0],  color=blue, label='DR-DDP')[0]
            
            ax.plot(x_mean[0,0,0], x_mean[0,1,0], marker='o', color=blue)[0]
            
            for obs_indx, obs in enumerate(obs_list[0]):
                obs_patch = ax.add_patch(plt.Circle((-100, -100), r_obs_list[0][obs_indx], facecolor=obs_face_color, edgecolor=obs_edge_color))
                obs_patch_list.append(obs_patch)
                obs_traj_plot = ax.plot(x_mean[0,3+2*obs_indx,0], x_mean[0,3+2*obs_indx+1,0],  color=orange, linestyle=None, marker='.', label='Obstacle', markersize=5)[0]
                obs_traj_plot_list.append(obs_traj_plot)
    
    
    
            plt.legend()
            # plt.grid(True)
            plt.axis([-5, 5, -5, 5])
            ax.set_aspect('equal', adjustable='box')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            fig.tight_layout()

            
            if render:
                FFwriter = matplotlib.animation.FFMpegWriter(bitrate=500)
                ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames=T)
                ani.save(path + 'video.mp4', writer=FFwriter)
            else:
                for frames in range(0,T):
                    update_graph(frames)
            
            plt.axis([-5, 5, -5, 5])
            ax.set_aspect('equal', adjustable='box')

            plt.savefig(path + 'final.pdf', dpi=300)
            
            plt.clf()
            
    
        except:
            col_mean, col_std = 0, 0
            col_minmax_ddp_mean, col_minmax_ddp_std = 0, 0
            col_ddp_mean, col_ddp_std = 0, 0
            pass
        
        plt.close('all')




    
    print('cost: {} ({})'.format(J_mean[0], J_std[0]) , 'time: {} ({})'.format(time_ar.mean(), time_ar.std()), 'No Collisions: {} ({})'.format(col_mean, col_std))

    if Baseline==1:
        print('cost_{}:{} ({})'.format(ddp_baseline_name, J_minmax_ddp_mean[0],J_minmax_ddp_std[0]), 'time_{}: {} ({})'.format(ddp_baseline_name, time_minmax_ddp_ar.mean(), time_minmax_ddp_ar.std()), 'No Collisions: {} ({})'.format(col_minmax_ddp_mean, col_minmax_ddp_std))
    elif Baseline==2:
        print('cost_{}:{} ({})'.format(ddp_baseline_name[0], J_minmax_ddp_mean[0],J_minmax_ddp_std[0]), 'time_{}: {} ({})'.format(ddp_baseline_name[0], time_minmax_ddp_ar.mean(), time_minmax_ddp_ar.std()), 'No Collisions: {} ({})'.format(col_minmax_ddp_mean, col_minmax_ddp_std))
        print('cost_{}:{} ({})'.format(ddp_baseline_name[1], J_ddp_mean[0],J_ddp_std[0]), 'time_{}: {} ({})'.format(ddp_baseline_name[1], time_ddp_ar.mean(), time_ddp_ar.std()), 'No Collisions: {} ({})'.format(col_ddp_mean, col_ddp_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--system', required=False, default="Car-v0", type=str) #simulation type (single or multiple)
    parser.add_argument('--baseline', required=False, default=2, type=int) #simulation type (single or multiple)
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--num_sim', required=False, default=1, type=int) #number of simulation runs

    args = parser.parse_args()

    print('\n-------Summary-------')

    path = "./results/{}/{}/".format(args.system, args.dist)
    

    DRDDP_file = open(path + 'wddp_1.pkl', 'rb')
    DDP_file = open(path + 'baseline.pkl', 'rb')
    output_wddp_list = pickle.load(DRDDP_file)
    output_baseline_list = pickle.load(DDP_file)
    DRDDP_file.close()
    DDP_file.close()

    if args.baseline==1:
        ddp_baseline_name = 'box-DDP'
    elif args.baseline==2:
        ddp_baseline_name = ['GT-DDP', 'box-DDP']
    else:
        ddp_baseline_name = 'None'



    summarize(output_baseline_list, output_wddp_list, args.dist, path, args.num_sim, ddp_baseline_name, args.plot)

