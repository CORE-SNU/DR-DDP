#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym

import torch
import numpy as np

import matplotlib.pyplot as plt
import DDP_Systems



class RacingCar(gym.Env):
    
    def __init__(self, dist, N_samples, num_sim, autograd=True, path_=None, render=False):
        
        self.autograd = autograd #Use automatic differentiation (if False, need to have analytic partial derivatives)
        self.path = path_  #Path to the folder
        self.nx = 5 #State dimension
        self.nu = 2 #Control input dimension
        self.nw = 2 #Disturbance dimension
        self.dist = dist #Disturbance distribution type (Currently implemented only for "normal" and "uniform")
        self.N_samples = N_samples #Number of disturbance samples
        self.num_sim = num_sim #Number of simulations
        self.goal = np.array([10, 2, 0])[...,np.newaxis] #Goal state to reach
        self.obstacle_1 = np.array([3, 0.5])[...,np.newaxis] #Position of the obstacle
        self.x0_car = np.array([-4.5, -0.5, 0])[...,np.newaxis] #Initial state of the car
        self.x0 = np.vstack([self.x0_car, self.obstacle_1]) #Initial state (includes both the car's and obstacle's states)
        self.obstacle = [self.obstacle_1] #A list of all obstacles
        self.num_obs = len(self.obstacle) #Number of obstacles
        self.safe_rad = 0.2 #Safety radius
        self.r_obs = [0.2] #Obstacle radius
        
        #------- Car Parameters -------
        self.L = 0.2 #Car Length
        self.delta_max = np.radians(30) #Maximum steering angle
        self.delta_min = -np.radians(30) #Minimum steering angle
        self.vel_max = 1.0 #Maximum velocity
        self.vel_min = -1.0 #Minimum velocity
        self.u_min = np.array([self.vel_min, self.delta_min]) #Control input upper limit
        self.u_max = np.array([self.vel_max, self.delta_max]) #Control input lower limit
        

        self.colors = ['red', 'blue'] #Colors for plotting the obstacles
        self.render_ = render #Render if true

        #-------Path Planning-------
        temp = np.arange(-4.5,4.5,0.05)
        path_x = np.concatenate([temp])
        path_y = np.concatenate([-0.5*np.ones(temp.shape[0])])
        path_yaw = np.concatenate([np.zeros(temp.shape[0])])
        r_circ = 1.5
        theta_circ = np.linspace(np.pi/2,np.pi, 15)
        self.xv_obs = np.concatenate([np.linspace(3, 1, temp.shape[0]//2-20), 1+r_circ*np.cos(theta_circ), -0.5*np.ones(temp.shape[0]//2+15)])
        self.yv_obs = np.concatenate([0.5*np.ones(temp.shape[0]//2-20), -1+r_circ*np.sin(theta_circ), np.linspace(-1, -4.5, temp.shape[0]//2+15)])
        self.ref_path = np.array([[path_x, path_y, path_yaw]]).T
        
        self.T = path_x.shape[0]-1 #Time horizon 
        self.T_sim = self.T   #Simulation time
        self.dt = 0.05 #Sampling rate
        
        #-------Cost Matrices-------
        self.Q = np.diag([10, 10, 1.]) #State deviatioon weight
        self.Qf = self.Q #Terminal stage state deviation weight
        self.R = np.diag([0.01, 0.01]) #Control input deviation weight
        self.obs_weight = 20. #Obstacle avoidance weight

        self.alpha = 1.0 #Initial line-search patameter
        
        
        #-------Set the penalty parameter depending on the distribution-------
        if self.dist=="normal":
            self.lambda_ = 1000
        elif self.dist=="uniform":
            self.lambda_ = 5000
       
        #-------Set distribution parameters-------
        if self.dist =="uniform":
            self.w_max = 0.01*np.ones(self.nw)
            self.w_min = -0.01*np.ones(self.nw)
            self.mu_w = (0.5*(self.w_max + self.w_min))[:,None]
            self.Sigma_w = 1/12*np.diag((self.w_max - self.w_min)**2)
        
        elif dist == "normal":
            self.w_max = None
            self.w_min = None
            self.mu_w = 0.01*np.ones((self.nw, 1))
            self.Sigma_w= 0.001*np.eye(self.nw)


        #-------Initialization-------
        self.u_init = np.zeros((self.T, self.nu, 1))
        self.w_init = np.zeros((self.T, self.nw, 1))
        self.x_init = np.zeros((self.T+1, self.nx, 1))
        self.x_init[0] = self.x0
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
        
        #Compute the cost and check for collisions
        cost = self.cost(x_, action[:self.nu], self.time_step)[0,0]
        col = self.check_col(x)
        
        #For terminal time stage
        done = False
        if self.time_step == self.T-1:
            col += self.check_col(x_)
            cost += self.terminal_cost(x_, self.time_step)
            done = True
        
        self.state = x_
        self.last_u = action[:self.nu]
        self.traj.append(self.state.copy())
        self.time_step += 1

        return x_, cost, done, {'obs': self.obstacle, 'r_obs': self.r_obs, 'col': col}
        
        
    def reset(self, seed=None, reset_sim=None, options=None):
        
        #If simulation has to be reset, generate new disurbance samples, true disturbances, and an empirical distribution
        if not reset_sim:
            np.random.seed(seed)
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
        self.mu_hat = 0*np.ones((self.T, self.nw, 1))

        #Generate true disturbance sequence for the entire simulation time
        self.true_w = np.zeros((self.num_sim, self.T_sim, self.nw, 1))
        for i in range(self.num_sim):
            for t in range(self.T_sim):
                if self.dist=="normal":
                    self.true_w[i,t] = self.normal(self.mu_w, self.Sigma_w)
                elif self.dist=="uniform":
                    self.true_w[i,t] = self.uniform(self.w_max, self.w_min)
                    
                    
    def dyn(self, states, u, w, t, grad=False):
        #The dynamics of a simple kinematic car
        
        x = states[0]
        y = states[1]
        theta = states[2]
        
        v = u[0]
        delta = u[1]
    
    
        if grad:
            tan_steer = torch.tan(delta)
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
        else:
            tan_steer = np.tan(delta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
        
        x_ = [x + self.dt*v*cos_theta,
              y + self.dt*v*sin_theta ,
              theta + self.dt*v/self.L*tan_steer,
              self.xv_obs[t] + w[0],
              self.yv_obs[t] + w[1]]
        
        if grad:
            x_ = torch.stack(x_)
        else:
            x_ = np.vstack(x_)
        return x_
    

    def check_col(self, x):
        #Checks for collision with the obstacles
        
        col = 0
        for i in range(self.num_obs):
            if ((x[:2]-x[3+2*i:3+2*(i+1)]).T @ (x[:2] - x[3+2*i:3+2*(i+1)]))**0.5 <= self.r_obs[i]+self.safe_rad:
                col += 1
        return col


    def cost(self, x, u, t, grad=False):
        #Calculates the running cost (control cost + tracking cost + obstacle avoidance cost)
        
        if grad:
            R = torch.from_numpy(self.R)
            Q = (torch.from_numpy(self.Q).type(torch.DoubleTensor))
            goal = torch.from_numpy(self.ref_path[t])
            cost_obs = 0.
            for i in range(self.num_obs):
                cost_obs += torch.exp(- (x[:2]-x[3+2*i:3+2*(i+1)]).T @ (x[:2] - x[3+2*i:3+2*(i+1)])/(2*(self.r_obs[i]+self.safe_rad)**2))

        else:
            R = self.R
            Q = self.Q
            goal = self.ref_path[t]
            
            cost_obs = 0.
            for i in range(self.num_obs):
                cost_obs += np.exp(- (x[:2]-x[3+2*i:3+2*(i+1)]).T @ (x[:2] - x[3+2*i:3+2*(i+1)])/(2*(self.r_obs[i]+self.safe_rad)**2))

        return u.T @ R @ u + (x[:3] - goal).T @ Q @ (x[:3] - goal) + self.obs_weight*cost_obs

    def terminal_cost(self, x, t, grad=False):
        #Calculates the terminal cost (tracking cost)

        if grad:
            Qf = (torch.from_numpy(self.Qf).type(torch.DoubleTensor))
            goal = torch.from_numpy(self.ref_path[t])
        else:
            Qf = self.Qf
            goal = self.ref_path[t]
        return (x[:3] - goal).T @ Qf @ (x[:3] - goal)
    
    
    
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
    