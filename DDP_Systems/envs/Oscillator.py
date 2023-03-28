#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym

import torch
import numpy as np

import DDP_Systems
from scipy import stats


class Oscillator(gym.Env):
    def __init__(self, dist, N_samples, num_sim, nx=10, autograd=True, path_=None, render=False):
        self.autograd = autograd #Use automatic differentiation (if False, need to have analytic partial derivatives)
        self.path = path_ #Path to the folder
        self.nx = nx #State dimension
        self.nu = 1 #Control input dimension
        self.nw = nx #Disturbance dimension
        self.dist = dist #Disturbance distribution type (Currently implemented only for "normal" and "uniform")
        self.N_samples = N_samples #Number of disturbance samples
        self.num_sim = num_sim #Number of simulations
        self.u_min = np.ones(self.nu) #Control input upper limit
        self.u_max = np.inf*np.ones(self.nu) #Control input lower limit
        
        self.render_ = render #Render if true
    
        self.T = 100 #Time horizon 
        self.T_sim = self.T #Simulation time
        self.ref_path = np.zeros((self.T, self.nx)) #Reference trajectory
        self.dt = 0.03 #Sampling rate
        self.rho = 0.0001 #Cost weight
        
        self.x0_all = stats.norm.pdf(np.linspace(0, np.pi, 100))[:, None] #Random initial states
        sigmaOm = 0.01
        omNum = 0.2*np.random.normal(0,sigmaOm, (100,1))
        self.omega = omNum - np.mean(omNum) #Oscillator frequency omega
        self.K = np.abs(np.max(self.omega) - np.min(self.omega)) + 1 #Oscillator coupling strength K
                
        
        self.alpha = 1.0 #Initial line-search patameter


        #-------Set the penalty parameter depending on the distribution-------
        if self.dist=="normal":
            self.lambda_ = 10000
        elif self.dist=="uniform":
            self.lambda_ = 1000
       
        
        #-------Set distribution parameters-------
        if self.dist =="uniform":
            self.theta = 0.01
            self.w_max = 0.1*np.ones(self.nw)
            self.w_min = -0.1*np.ones(self.nw)
            self.mu_w = (0.5*(self.w_max + self.w_min))[:,None]
            self.Sigma_w = 1/12*np.diag((self.w_max - self.w_min)**2)
        
        elif dist == "normal":
            self.theta = 0.01
            self.w_max = None
            self.w_min = None
            self.mu_w = 0.001*np.ones((self.nw, 1))
            self.Sigma_w= 0.001*np.eye(self.nw)


        #-------Initialization-------
        self.u_init = np.zeros((self.T, self.nu, 1))
        self.w_init = np.zeros((self.T, self.nw, 1))
        self.x_init = np.zeros((self.T+1, self.nx, 1))
        self.samples = None
        self.mu_hat = None
        self.Sigma_hat = None
        self.true_w = None
        self.traj = []
        self.state = None
    
    def step(self, action):
        x = self.state.copy()
        
        #Apply the control input and disturbance to the system
        x_ = self.dyn(x, action[:self.nu], action[self.nu:], self.time_step)
        
        #Compute the cost
        cost = self.cost(x_, action[:self.nu], self.time_step)
        
        #For terminal time stage
        done = False
        if self.time_step == self.T-1:
            cost += self.terminal_cost(x_, self.time_step)
            done = True
        
        self.state = x_
        self.last_u = action[:self.nu]
        self.traj.append(self.state.copy())
        self.time_step += 1

        return x_, cost, done, {}
        
        
    def reset(self, seed=None, reset_sim=None, options=None):
        
        #If simulation has to be reset, choose new random initial states, generate new disurbance samples, true disturbances, and an empirical distribution
        if not reset_sim:
            self.x0 = self.x0_all[:self.nx]
            self.x_init[0] = self.x0
            self.gen_dist()
        
        else:
            self.state = self.x0.copy()
            self.traj = []
            self.traj.append(self.state.copy())
            self.time_step = 0

        return self.state, {}
    
    def gen_dist(self):
        
        #Generate N_sample disturbance samples and the mean vector and covariance matrix of the empirical distribution
        self.samples, self.mu_hat, self.Sigma_hat = self.gen_sample_dist(self.dist, self.T, self.N_samples, mu_w=self.mu_w, Sigma_w=self.Sigma_w, w_max=self.w_max, w_min=self.w_min)

        #Generate true disturbance sequence for the entire simulation time
        self.true_w = np.zeros((self.num_sim, self.T_sim, self.nw, 1))
        for i in range(self.num_sim):
            for t in range(self.T_sim):
                if self.dist=="normal":
                    self.true_w[i,t] = self.normal(self.mu_w, self.Sigma_w)
                elif self.dist=="uniform":
                    self.true_w[i,t] = self.uniform(self.w_max, self.w_min)
                    
    def dyn(self, x, u, w, t, grad=False):
        #The dynamics of a coupled oscillators (Kuramato model)
        
        x_ = []

        if grad:
            omega = torch.from_numpy(self.omega)
            for i in range(self.nx):
                x_new = x[i] + self.dt*(omega[i] + self.K*u[0]/self.nx*torch.sum(x - x[i]) + w[i])
                x_.append(x_new)
            
            x_ = torch.stack(x_)
        
        else:
            for i in range(self.nx):
                x_new = x[i] + self.dt*(self.omega[i] + self.K*u[0]/self.nx*np.sum(x - x[i]) + w[i])
                x_.append(x_new)
                
            x_ = np.vstack(x_)
        
        return x_
    
    def cost(self, x, u, t, grad=False):
        #Calculates the running cost
        cost = self.rho*u.T @ u
        if grad:
            for i in range(self.nx):
                cost += torch.sum(torch.sin(x[i] - x)**2)
        else:
            for i in range(self.nx):
                cost += np.sum(np.sin(x[i] -x)**2)
        return cost

    def terminal_cost(self, x, t, grad=False):
        #Calculates the terminal cost

        cost = 0
        if grad:
            for i in range(self.nx):
                cost += torch.sum(torch.sin(x[i] - x)**2)
        else:
            for i in range(self.nx):
                cost += np.sum(np.sin(x[i] -x)**2) 
        return cost
    
    
    
    def dyn_grad(self, xuw, x):
        #Computes the partial derivatives of the dynamics

        d_dxuw = self.jacobian(x, xuw)
        d_dx = d_dxuw[:, :self.nx].detach().numpy()[:,:,0]
        d_du = d_dxuw[:, self.nx:self.nx+self.nu].detach().numpy()[:,:,0]
        d_dw = d_dxuw[:, self.nx+self.nu:].detach().numpy()[:,:,0]
 
        return d_dx, d_du, d_dw
    
    def cost_grad(self, x, u, t):
        #Computes the partial derivatives of the running cost

        xu = torch.cat([x, u])
        l = self.cost(xu[:self.nx], xu[self.nx:], t, grad=True)
        
        l_xu, = torch.autograd.grad(l, xu, create_graph=not False)
        l_x = l_xu[:self.nx].detach().numpy()[..., np.newaxis][:,:,0]
        l_u = l_xu[self.nx:].detach().numpy()[..., np.newaxis][:,:,0]
        l_xuxu = self.jacobian(l_xu, xu)
        l_xx = l_xuxu[:self.nx, :self.nx].detach().numpy()[:,:,0]
        l_ux = l_xuxu[self.nx:, :self.nx].detach().numpy()[:,:,0]
        l_uu = l_xuxu[self.nx:, self.nx:].detach().numpy()[:,:,0]
        
        return l_x, l_u, l_xx, l_ux, l_uu
    
    def terminal_cost_grad(self, x, t):
        #Computes the partial derivatives of the terminal cost

        l = self.terminal_cost(x, t, grad=True)
        l_x, = torch.autograd.grad(l, x, create_graph=not False)
        l_xx = self.jacobian(l_x, x).detach().numpy()[:,:,0]
        l_x = l_x.detach().numpy()[..., np.newaxis][:,:,0]
       
        return l_x, l_xx
        
        
    def jacobian(self, y, x, **kwargs):
        J = [self.grad(y[i], x, **kwargs) for i in range(y.shape[0])]
        J = torch.stack(J)
        J.requires_grad_()
        return J
    
    def grad(self, y, x, allow_unused=True, **kwargs):
        dy_dx, = torch.autograd.grad(
            y, x, retain_graph=True, allow_unused=allow_unused, **kwargs)
    
        # The gradient is None if disconnected.
        dy_dx = dy_dx if dy_dx is not None else torch.zeros_like(x)
        dy_dx.requires_grad_()
    
        return dy_dx
    
    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T
    
    def normal(self, mu, Sigma, N=1):
        x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
        return x


    def gen_sample_dist(self, dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
        if dist=="normal":
            sample = self.normal(mu_w, Sigma_w, N=(T, N_sample))
        elif dist=="uniform":
            sample = self.uniform(w_max, w_min, N=N_sample*T)
            sample = np.reshape(sample, (self.nw, N_sample, T))
        
        mean_ = np.average(sample, axis = 1)
        nw = mean_.shape[0]
        
        var_ = np.zeros((nw, nw, T))
        for t in range(T):
            var_[:,:,t] = np.cov(sample[:,:,t])
        return sample.T, mean_.T[...,np.newaxis], var_.transpose(2,0,1)
    
    
    def render(self, mode='human', message=None):
        pass
    
    def close(self):
        pass
    

