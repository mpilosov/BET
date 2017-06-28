# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
import numpy as np

    
def my_model(parameter_samples):
    # model y(t) = a*t^n + b*t + c. unknown parameter lambda = [a, b, c, n]
    # rows are samples from parameter space lambda, columns are a,b,c,n
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    weights = 1.0/1.0
    QoI_samples = np.array([A[:,0]*(t**A[:,3]) + A[:,1]*t + A[:,2] for t in range(1,5)]).transpose()
    return np.sum( ( (QoI_samples - np.array([1,2,3,4]) )**2 )*weights , 1) # RSS 

def my_model2(parameter_samples):
    # model y(t) = a*t + c. unknown parameter lambda = [a, c]
    # rows are samples from parameter space lambda, columns are a and c
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*t + A[:,1] for t in range(1,5)]).transpose()
    return QoI_samples

def my_model3(parameter_samples):
    # model y(t) = a*t^n + b*t + c. unknown parameter lambda = [a, b, c, n]
    # rows are samples from parameter space lambda, columns are a,b,c,n
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*(t**A[:,2]) + A[:,1]*t  for t in range(1,5)]).transpose()
    return QoI_samples
    
def my_model4(parameter_samples):
    # model y(t) = a*t^n + b*t + c. unknown parameter lambda = [a, b, c, n]
    # rows are samples from parameter space lambda, columns are a,b,c,n
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*(t**A[:,3]) + A[:,1]*t + A[:,2] for t in range(1,5)]).transpose()
    return QoI_samples

def griewank(parameter_samples):
    n,d = parameter_samples.shape # num_samples, dimension
    s = np.zeros(n) # sum
    p = np.ones(n) # prod
    for i in range(d):
        s += parameter_samples[:,i]**2
        p *= np.cos(parameter_samples[:,i]/np.sqrt(i+1))
    return 1 + s/4000. - p