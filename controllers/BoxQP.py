#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.linalg import lstsq

def boxQP(H, g, lower, upper, x0=None, options=None, nargout=4):
    n = H.shape[0]
    clamped = np.zeros((n,1))
    free = np.ones((n,1))
    oldvalue = 0
    result = 0
    gnorm = 0
    nfactor = 0
    trace = []
    Hfree = np.zeros((n, n))

    if x0 is not None and x0.shape[0] == n:
        x = np.clip(x0, lower, upper)
    else:
        LU = np.array([lower, upper])
        LU[np.logical_not(np.isfinite(LU))] = np.nan
        x = np.nanmean(LU, axis=0)
    x[np.logical_not(np.isfinite(x))] = 0

    if options is not None:
        maxIter, minGrad, minRelImprove, stepDec, minStep, Armijo, print_ = options
    else:
        maxIter = 100
        minGrad = 1e-8
        minRelImprove = 1e-8
        stepDec = 0.6
        minStep = 1e-22
        Armijo = 0.1
        print_ = 0

#    value = x.T @ g + 0.5 * x.T @ H @ x
    value = np.sum(x*g) + 0.5 * np.sum(x[:,0]*np.sum(H.T*x, axis=0))
    
    if print_ > 0:
        print('==========\nStarting box-QP, dimension {}, initial value: {:3f}'.format(n, value))

    for iter_ in range(1, maxIter+1):
        if result != 0:
            break
        if iter_ > 1 and (oldvalue - value) < minRelImprove * abs(oldvalue):
            result = 4
            break
        oldvalue = value

        grad = g + np.sum(H.T*x, axis=0, keepdims=1).T


        old_clamped = clamped
        #clamped = np.full((n,1), False)
        clamped = np.zeros((n,1))
        clamped[(x == lower) & (grad > 0)] = 1
        clamped[(x == upper) & (grad < 0)] = 1
        free = np.logical_not(clamped).flatten()

        if all(clamped):
            result = 6
            break

        if iter_ == 1:
            factorize = True
        else:
            factorize = np.any(old_clamped != clamped)

        if factorize:
            try:
                Hfree = np.linalg.cholesky(H[np.ix_(free, free)])
            except np.linalg.LinAlgError:
                result = -1
                break
            nfactor += 1

        gnorm = ((grad[free]**2).sum())**0.5 #np.linalg.norm(grad[free])
        if gnorm < minGrad:
            result = 5
            break

        grad_clamped = g + np.sum(H.T*(x*clamped), axis=0, keepdims=1).T # H @ (x * clamped)
        
        search = np.zeros((n,1))
        search[free] = - lstsq(Hfree.T, lstsq(Hfree, grad_clamped[free], lapack_driver='gelsy')[0], lapack_driver='gelsy')[0] - x[free]
        
        sdotg = sum(search * grad)
        if sdotg >= 0:
            break
        
        step = 1
        nstep = 0
        xc = np.clip(x + step*search, lower, upper)
        #np.array([min(upper[i], max(lower[i], x[i] + step * search[i])) for i in range(n)])
        
#        vc = xc.T @ g + 0.5 * xc.T @ H @ xc

        vc = np.sum(xc*g) + 0.5 * np.sum(xc[:,0]*np.sum(H.T*xc, axis=0))
        
        while (vc - oldvalue) / (step * sdotg) < Armijo:
            step *= stepDec
            nstep += 1
            xc = np.clip(x + step*search, lower, upper)
            vc = np.sum(xc*g) + 0.5 * np.sum(xc[:,0]*np.sum(H.T*xc, axis=0))
            if step < minStep:
                result = 2
                break
            
        if print_ > 1:
            print('iter {:3d}  value {:-9.5g} |g| {:-9.3g}  reduction {:-9.3g}  linesearch {:g}^{:2d}  n_clamped {}'.format(iter_, vc, gnorm, oldvalue-vc, stepDec, nstep, sum(clamped.flatten())))
        
        if nargout > 4:
            trace.append({'x': x, 'xc': xc, 'value': value, 'search': search, 'clamped': clamped, 'nfactor': nfactor})
        
        # accept candidate
        x = xc
        value = vc
                
        
    if iter_ >= maxIter:
        result = 1
    
    results = [
        'Hessian is not positive definite',
        'No descent direction found',
        'Maximum main iterations exceeded',
        'Maximum line-search iterations exceeded',
        'Improvement smaller than tolerance',
        'No bounds, returning Newron point',
        'Gradient norm smaller than tolerance',
        'All dimensions are clamped'
    ]
    
    if print_ > 0:
        print(f'RESULT: {results[result + 1]}.\niterations {iter_}  gradient {gnorm} final value {value}  factorizations {nfactor}')
    
    if result < 1:
        print(result)
        
    if nargout <= 4:
        return x,result,Hfree,free
    else:
        return x,result,Hfree,free, trace
#
#def clamp(x, lower, upper):
#    return np.maximum(lower, np.minimum(upper, x))
    
def demoQP():
    np.random.seed(1337)
    options = [100, 1e-8, 1e-8, 0.6, 1e-22, 0.1, 1] # defaults with detailed printing
    n = 500
    for i in range(5):
        g = np.random.rand(n, 1)
        H = np.random.rand(n, n)
        H = H @ H.T
        lower = -np.ones((n, 1))
        upper = np.ones((n, 1))
        tic = time.time()
        x,result,Hfree,free,trace = boxQP(H, g, lower, upper, np.random.rand(n, 1), options)
        toc = time.time() - tic
        print("Elapsed time: ", toc)
    return 
    
def main():
    demoQP()
    
if __name__ == '__main__':
    main()
    

    