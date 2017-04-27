"""Temperature variations of a thin (1-D) rod with a heat source 'underneath' and perfectly insulated (Neumann boundaries set to 0).  We turn the heat source off after time t_heatoff"""

from dolfin import *
import numpy as np
from simulation_2kappas_setup import *


def heatROD(i, amp, px, width, degree, T_R, kappa_0, kappa_1, rho, cap, nx, mesh, dt, t_stop, saveopt=True):

    #define the subspace we will solve the problem in
    V = FunctionSpace(mesh, 'Lagrange', 1)

    #turn off the heat halfway through
    t_heatoff = t_stop/2.0

    #time stepping method. forward, backward, CN, etc...
    theta = 1
    
    #split the domain down the middle(dif therm cond)
    kappa_str = 'x[0] > 0.5 ?'\
                   'kappa_1 : kappa_0'

    # Physical parameters
    kappa = Expression(kappa_str, kappa_0=kappa_0, kappa_1=kappa_1, degree=1)

    # Define initial condition(initial temp of plate)
    T_1 = interpolate(Constant(T_R), V)

    # Define variational problem
    T = TrialFunction(V)

    #two f's and L's for heat source on and off
    f_heat = Expression('amp*exp(-(x[0]-px)*(x[0]-px)/width)', amp=amp, px=px, width=width, degree=1)
    #plot(interpolate(f_heat, V))
    #interactive()
    f_cool = Constant(0)
    v = TestFunction(V)
    a = rho*cap*T*v*dx + theta*dt*kappa*inner(nabla_grad(v), nabla_grad(T))*dx
    L_heat = (rho*cap*T_1*v + dt*f_heat*v - (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(T_1)))*dx
    L_cool = (rho*cap*T_1*v + dt*f_cool*v - (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(T_1)))*dx
    
    A = assemble(a)
    b = None  # variable used for memory savings in assemble calls

    T = Function(V)
    t = dt
   
    #time stepping
    while t <= t_stop:
    # while t <= 0.99:
        #plot(T)
        #interactive()
        relerrorvec = []
        #print 'time =', t
        if t < t_heatoff:
            b = assemble(L_heat, tensor=b)
        else:
            b = assemble(L_cool, tensor=b)
        solve(A, T.vector(), b)

        t += dt
        T_1.assign(T)

    # print t
    if saveopt == True:    
        filename = 'Tfiles/Tsample_' + str(i) + '.xml'
        file = File(filename)
        file << T
    
       
    # plot(T)
    # interactive()
    return T




