# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
import numpy as np

    
def my_model(parameter_samples):
    # model y(t) = a*t^n + b*t + c. unknown parameter lambda = [a, b, c, n]
    # rows are samples from parameter space lambda, columns are a,b,c,n
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*(t**A[:,3]) + A[:,1]*t + A[:,2] for t in range(1,5)]).transpose()
    return QoI_samples

def my_model2(parameter_samples):
    # model y(t) = a*t + c. unknown parameter lambda = [a, c]
    # rows are samples from parameter space lambda, columns are a and c
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*t + A[:,1] for t in range(1,5)]).transpose()
    return QoI_samples

def my_model4(parameter_samples):
    # model y(t) = a*t^n + b*t + c. unknown parameter lambda = [a, b, c, n]
    # rows are samples from parameter space lambda, columns are a,b,c,n
    # our model is evaluated at t = [1,2,3,4], which are the four cols of output
    A = parameter_samples
    QoI_samples = np.array([A[:,0]*(t**A[:,3]) + A[:,1]*t + A[:,2] for t in range(1,5)]).transpose()
    return QoI_samples
