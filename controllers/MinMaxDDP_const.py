#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import torch 
from .BoxQP import *

class MinMaxDDP_const:
    def __init__(self, system, wc, path=None, seed=1337):
        self.seed = seed
        self.path = path
        self.system = system
        self.dist = system.dist
        self.T = system.T
        self.samples = None
        self.true_w = None
        self.N_samples = system.N_samples
        self.nx = system.nx
        self.nu = system.nu
        self.nw = system.nw
        self.mu_hat = None
        self.Sigma_hat = None
        self.mu_w = system.mu_w
        self.Sigma_w = system.Sigma_w
        self.x0 = None
        self.wc = wc
        if self.dist=="uniform":
            self.w_max = system.w_max
            self.w_min = system.w_min
  
        
        self.A = np.zeros((self.T, self.nx, self.nx))
        self.B = np.zeros((self.T, self.nx, self.nu))
        self.Xi = np.zeros((self.T, self.nx, self.nw))
        self.d = np.zeros((self.T, self.nx, 1))
        self.Q = np.zeros((self.T+1, self.nx, self.nx))
        self.R = np.zeros((self.T, self.nu, self.nu))
        self.S = np.zeros((self.T, self.nu, self.nx))
        self.q = np.zeros((self.T+1, self.nx, 1))
        self.s = np.zeros((self.T, self.nu, 1))
        self.c = np.zeros(self.T+1)
        self.P = np.zeros((self.T+1, self.nx, self.nx))
        self.r = np.zeros((self.T+1, self.nx, 1))
        self.z = np.zeros((self.T+1, 1, 1))
        self.K = np.zeros(( self.T, self.nu, self.nx))
        self.L = np.zeros(( self.T, self.nu, 1))
        self.H = np.zeros(( self.T, self.nw, self.nx))
        self.h = np.zeros(( self.T, self.nw, 1))
        self.alpha = system.alpha
        self.tol = 1e-06
        self.n_iterations = 1000
        self.u_max = system.u_max[:, None]
        self.u_min = system.u_min[:, None]
        self.reg_lambda = 1e-5
        self.regType = 1
        self.dlambda = 1
        self.lambdaMin = 1e-6
        self.lambdaMax = 1e10
        self.lambdaFactor = 1.6
    
    def compute_deriv(self, xuw, t):
        #Compute all partial derivatives at time t

        if t == self.T:
            x_bar = xuw[t][:self.nx]
            l_x, l_xx = self.system.terminal_cost_grad(x_bar, t)
            self.Q[t] = l_xx
            self.q[t] = l_x
            self.c[t] = self.system.terminal_cost(x_bar, t, grad=True).detach().numpy()
        else:
            x_bar = xuw[t][:self.nx]
            u_bar = xuw[t][self.nx:self.nx+self.nu]
            w_bar = xuw[t][self.nx+self.nu:]
            x_bar_ = xuw[t+1][:self.nx]
            self.d[t] = x_bar_.detach().numpy()
            A, B, Xi = self.system.dyn_grad(xuw[t], x_bar_)
    
            self.A[t], self.B[t], self.Xi[t] = A, B, Xi
            
            l_x, l_u, l_xx, l_ux, l_uu = self.system.cost_grad(x_bar, u_bar, t)
            self.Q[t] = l_xx
            self.q[t] = l_x
            self.R[t] = l_uu
            self.s[t] = l_u
            self.S[t] = l_ux
            self.c[t] = self.system.cost(x_bar, u_bar, t, grad=True).detach().numpy()
    
    def riccati(self, P, r, z, t, x_bar, u_bar, w_bar, w_bar_hat, Sigma_hat, w_hat):
        #Update the parameters of the value function and compute the control and disturbance policies

        u_bar_np = u_bar.detach().numpy()

        Theta_X = self.A[t].T @ r + self.q[t]
        Theta_U = self.B[t].T @ r + self.s[t]
        Theta_W = self.Xi[t].T @ r
        Theta_XX = self.A[t].T @ P @ self.A[t] + self.Q[t]
        Theta_UU = self.B[t].T @ P @ self.B[t] + self.R[t] 
        Theta_WW = self.Xi[t].T @ P @ self.Xi[t]
        Theta_XU = self.A[t].T @ P @ self.B[t] + self.S[t].T
        Theta_XW = self.A[t].T @ P @ self.Xi[t]
        Theta_UW = self.B[t].T @ P @ self.Xi[t]
        Theta_WU = Theta_UW.T
        Theta_UX = Theta_XU.T
        Theta_WX = Theta_XW.T
      
        L_init = - np.linalg.solve(Theta_UU - Theta_UW @ np.linalg.solve(Theta_WW, Theta_WU), Theta_U - Theta_UW @ np.linalg.solve(Theta_WW, Theta_W) )
        
   
        Theta_UU_reg = self.B[t].T @ P @ self.B[t] + self.R[t] + self.reg_lambda*np.eye(self.nu)*(self.regType == 1)
        Theta_WW_reg = Theta_WW + self.reg_lambda*np.eye(self.nw)*(self.regType == 1)

        lower = self.u_min - u_bar_np
        upper = self.u_max - u_bar_np
        
        #Solve a constrained QP problem
        options = [1000, 1e-8, 1e-8, 0.6, 1e-22, 0.1, 0]
        L, result, R, free = boxQP(Theta_UU_reg - Theta_UW @ np.linalg.solve(Theta_WW, Theta_WU), Theta_U - Theta_UW @ np.linalg.solve(Theta_WW, Theta_W), lower, upper, L_init, options)
        
        if result < 1:
            print('Diverged!')
            return True
            
        K = np.zeros((self.nu, self.nx))
        if np.any(free):
            Lfree = - np.linalg.solve(R, np.linalg.solve(R.T, (Theta_UX - Theta_UW @ np.linalg.solve(Theta_WW, Theta_WX) )[free, :]))
            K[free,:] = Lfree
            
        H = - np.linalg.solve(Theta_WW_reg, Theta_WX + Theta_WU @ K )
        h = - np.linalg.solve(Theta_WW_reg, Theta_W + Theta_WU @ L )
        
        P_ = Theta_XX + K.T @ Theta_UX + Theta_XU @ K + H.T @ Theta_WX + Theta_XW @ H + K.T @ Theta_UU @ K + H.T @ Theta_WW @ H + K.T @ Theta_UW @ H + H.T @ Theta_WU @ K
        r_ = Theta_X + K.T @ Theta_U + H.T @ Theta_W + Theta_XU @ L + Theta_XW @ h + K.T @ Theta_UU @ L + H.T @ Theta_WW @ h + K.T @ Theta_UW @ h + H.T @ Theta_WU @ L
        z_ = z + self.c[t] + L.T @ Theta_U + h.T @ Theta_W + 0.5*(L.T @ Theta_UU @ L + h.T @ Theta_WW @ h + L.T @ Theta_UW @ h + h.T @ Theta_WU @ L)
        
        return P_, r_, z_, K, L, H, h


    def backward(self, xuw, x_bar, u_bar, w_bar, w_bar_np, w_bar_hat, Sigma_hat, w_hat):
        #Perform the backward pass

        done = False
        while not done:
            self.compute_deriv(xuw, self.T)
            self.P[self.T] = self.Q[self.T]
            self.r[self.T] = self.q[self.T]
            self.z[self.T] = self.c[self.T]
            
    
            for t in range(self.T-1, -1, -1):
                self.compute_deriv(xuw, t)
                out = self.riccati(self.P[t+1], self.r[t+1], self.z[t+1], t, x_bar[t], u_bar[t], w_bar_np[t], w_bar_hat[t], Sigma_hat[t], w_hat[t])
            
                if out is True:
                    diverge = True
                    break
                else:
                    self.P[t], self.r[t], self.z[t], self.K[t], self.L[t], self.H[t], self.h[t] = out
                    diverge = False
                    
            if diverge:
                self.dlambda = np.max([self.dlambda * self.lambdaFactor, self.lambdaFactor])
                self.reg_lambda = np.max([self.reg_lambda*self.dlambda, self.lambdaMin])
                if self.reg_lambda > self.lambdaMax:
                    break
                pass
            done = True
            
    def simulate(self, render, outputs, sim_ind):
        #Apply the control input to the system

        alpha, u_bar, x_bar, w_bar, vf_list, J_list, time_ = outputs

        x = np.zeros((self.T+1, self.nx, 1))
        u = np.zeros((self.T, self.nu, 1))
        w = np.zeros((self.T+1, self.nw, 1))
        
        x[0] = self.x0x0
        J = 0.0
        t = 0
        done = False
        self.system.reset(reset_sim=True)
        while not done:

            u[t] = u_bar[t] + alpha*self.L[t] + self.K[t] @ (x[t] - x_bar[t]) 
            w[t] = self.true_w[sim_ind, t]
            action = np.concatenate([u[t], w[t]])
            
            x[t+1], cost_, done, info = self.system.step(action)

            if info!={}:
                if t==0:
                    obs_list = []
                    r_obs_list = []
                    col = 0
                obs_list.append(info['obs'])
                r_obs_list.append(info['r_obs'])
                col += info['col']
                
            if render:
                self.system.render(message=self.path+'MinMaxDDP')

            J += cost_
            t += 1

        self.system.close()
        
        if info !={}:
            return {'state_traj': x,
                    'control_traj': u,
                    'cost': J,
                    'vf': vf_list,
                    'J_wc': J_list,
                    'ref': self.system.ref_path,
                    'obs': obs_list,
                    'r_obs': r_obs_list,
                    'col': col,
                    'comp_time': time_}
        else:
            return {'state_traj': x,
                    'control_traj': u,
                    'cost': J,
                    'vf': vf_list,
                    'J_wc': J_list,
                    'ref': self.system.ref_path,
                    'comp_time': time_}

        
    def _trajectory_cost(self, x, u):
        #Compute the trajectory cost

        J = self.system.terminal_cost(x[self.T], self.T)
        for t in range(self.T-1, -1, -1):
            J = J + self.system.cost(x[t], u[t], t)
        return J
    
    def forward(self, alpha,  x, u, w):
        #Perform the forward pass and update the nominal trajectories

        x_new = torch.zeros_like(x)
        u_new = torch.zeros_like(u)
        w_new = torch.zeros_like(w)
        xuw_list = []
        x_new[0] = x[0]

        L = torch.from_numpy(self.L)
        K = torch.from_numpy(self.K)
        h = torch.from_numpy(self.h)
        H = torch.from_numpy(self.H)
        for t in  range(self.T):
            u_new[t] = u[t] + alpha * L[t] + K[t] @ (x_new[t] - x[t])
            w_new[t] = w[t] + alpha * h[t] + H[t] @ (x_new[t] - x[t])
            xuw = torch.cat([x_new[t], u_new[t], w_new[t]])
            x_new[t+1] = self.system.dyn(xuw[:self.nx], xuw[self.nx:self.nx+self.nu], xuw[self.nx+self.nu:], t, grad=True)
            xuw_list.append(xuw)
        
        xuw_list.append(x_new[t+1])
        return x_new, u_new, w_new, xuw_list
    
    def init_control(self, x, u, w):
        #Generate initial nominal trajectories

        x_new = torch.zeros((self.T+1, self.nx, 1), dtype=torch.double)
        x_new[0] = x[0]
        xuw_list = []
        
        for t in  range(self.T):
            xuw = torch.cat([x_new[t], u[t], w[t]])
#            w_new[t] = self.true_w[sim_ind, t]
            x_new[t+1] = self.system.dyn(xuw[:self.nx], xuw[self.nx:self.nx+self.nu], xuw[self.nx+self.nu:], t, grad=True)
            # x_new[t+1].requires_grad_()
            xuw_list.append(xuw)
        xuw_list.append(x_new[t+1])
        return x_new, u, w, xuw_list        
    
   
   
    def run(self, sim_ind):
        #Run the algorithm

        self.system.reset(seed=self.seed)
        self.mu_hat = self.system.mu_hat
        self.Sigma_hat = self.system.Sigma_hat
        self.samples = self.system.samples
        self.true_w = self.system.true_w
        self.x0 = self.system.x0.copy()
        
        vf_list = []
        J_list = []
        us_np = self.system.u_init
        ws_np = self.system.w_init
        xs_np = self.system.x_init
        xs = torch.from_numpy(xs_np)
        us = torch.from_numpy(us_np)
        ws = torch.from_numpy(ws_np)
        xs.requires_grad_()
        us.requires_grad_()
        ws.requires_grad_()
        converged = False
        alpha = self.alpha
        
        time_per_iter = []
        for iteration in range(self.n_iterations):
            start = time.time()
            if iteration == 0:
                #----- Generate initial nominal trajectories  -----
                xs, us, ws, xuw = self.init_control(xs, us, ws)
                J_opt = np.inf
                V_opt = np.inf
    
            #----- Backward Pass -----
            self.backward(xuw, xs, us, ws, ws_np, self.mu_hat, self.Sigma_hat, self.samples[..., np.newaxis])

            #----- Forward Pass  -----
            xs_new, us_new, ws_new, xuw_new = self.forward(alpha, xs, us, ws)
            xs_new_np, us_new_np, ws_new_np = xs_new.detach().numpy(), us_new.detach().numpy(), ws_new.detach().numpy()

            #----- Perform line search  -----
            J_new = self._trajectory_cost(xs_new_np, us_new_np)
            V = (self.x0 - xs_np[0]).T @ self.P[0] @ (self.x0 - xs_np[0]) + 2*self.r[0].T @ (self.x0 - xs_np[0]) + self.z[0]

            while J_new > J_opt:
                if alpha <= 1e-8:
                    J_new = J_opt
                    xs_new = xs
                    us_new = us
                    ws_new = ws
                    xs_new_np = xs_np
                    ws_new_np = ws_np
                    us_new_np = us_np
                    xuw_new = xuw
                    
                    converged = True
                    break
                
                alpha = alpha*0.1
                xs_new, us_new, ws_new, xuw_new = self.forward(alpha, xs, us, ws)
                xs_new_np, us_new_np, ws_new_np = xs_new.detach().numpy(), us_new.detach().numpy(), ws_new.detach().numpy()

                J_new = self._trajectory_cost(xs_new_np, us_new_np)
                V = (self.x0 - xs_np[0]).T @ self.P[0] @ (self.x0 - xs_np[0]) + 2*self.r[0].T @ (self.x0 - xs_np[0]) + self.z[0]

            #----- Terminate when the cost converges  -----
            if np.abs((J_opt - J_new)/J_opt) < self.tol:
                converged = True
                
            V_opt = V
            J_opt = J_new
            xs = xs_new
            us = us_new
            ws = ws_new
            xs_np = xs_new_np
            us_np = us_new_np
            ws_np = ws_new_np
            xuw = xuw_new

            end = time.time()
            time_ = end-start
            time_per_iter.append(time_)
        
            print('      iter:', iteration, ' J:', J_new[0,0], ' Vf:', V[0,0])
            vf_list.append(V[0,0])
            J_list.append(J_new[0,0])
            
            if converged:
                break

        
        return alpha, us_np, xs_np, ws_np, vf_list, J_list, time_per_iter
    

                
            
            
