
"""Temperature variations of a thin (2-D) plate with a heat source 'underneath' and perfectly insulated (Neumann boundaries set to 0).  We turn the heat source off after time t_heatoff"""

from dolfin import *
import numpy
from simulation_2kappas_setup import *


def heatplatecenter(i, amp, px, py, width, degree, T_R, kappa_0, kappa_1, rho, cap, nx, ny, mesh, sensor_points, dt, t_stop, data):

    data_vec = []

    #define the subspace we will solve the problem in
    V = FunctionSpace(mesh, 'Lagrange', 1)

    #turn off the heat halfway through
    t_heatoff = t_stop#/2.0

    #time stepping method. forward, backward, *CN* (0.5), etc...
    theta = 0.5

    #split the domain down the middle(dif therm cond)
    kappa_str = 'x[0] > 0 ?'\
                   'kappa_1 : kappa_0'

    # Physical parameters
    # ***DEGREE NEEDS TO EB CHANGED FOR NON-CONSTANT EXPRESSION***
    kappa = Expression(kappa_str, kappa_0=kappa_0, kappa_1=kappa_1, degree=0)

    # Define initial condition(initial temp of plate)
    T_1 = interpolate(Constant(T_R), V)

    # Define variational problem
    T = TrialFunction(V)

    #two f's and L's for heat source on and off
    f_heat = Expression('amp*exp(-((x[1]-py)*(x[1]-py)+(x[0]-px)*(x[0]-px))/width)', amp=amp, px=px, py=py, width=width, degree=4)
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
    count_t = 0
    while t <= t_stop:
        count_t += 1 
        relerrorvec = []
        #print 'time =', t
        if t < t_heatoff:
            b = assemble(L_heat)#, tensor=b)
        else:
            b = assemble(L_cool)#, tensor=b)
        solve(A, T.vector(), b)

        t += dt
        T_1.assign(T)

        # gather data every 50 time steps
        if np.mod(count_t, 50) == 0:
            #print 'IN!!!!!'
            for j in range(num_sensors):
                data_vec.append(T(sensor_points[j, :]))
        

    data[i, :] = data_vec


    #plot(T)
    #interactive()





