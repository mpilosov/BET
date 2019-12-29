from dolfin import *

def make_model(temp_locs_list):
    # heatrod code here - including all settings
    t_stop = 1.0

    # Some fixed parameter values
    amp = 50.0  # amplitude of the heat source
    px = 0.5  # location of the heat source
    width = 0.05  # width of the heat source
    T_R = 0  # initial temp of the plate
    cap = 1.5  # heat capacity
    rho = 1.5  # density

    # 'parameters' reserved for FEnICS
    parameters['allow_extrapolation'] = True

    # mesh properties
    nx = 50  # this is ~ our h value
    mesh = IntervalMesh(nx, 0, 1)
    degree = 1
    r = 1.0  # time stepping ratios - attention to stability
    dt = r/nx

    # turn off the heat halfway through
    t_heatoff = t_stop/2.0

    def my_model(parameter_samples):
        QoI_samples = np.zeros((len(parameter_samples), len(temp_locs_list)))

        for i in range(len(parameter_samples)):
            kappa_0 = parameter_samples[i, 0]
            kappa_1 = parameter_samples[i, 1]

            # define the subspace we will solve the problem in
            V = FunctionSpace(mesh, 'Lagrange', degree)

            # split the domain down the middle(dif therm cond)
            kappa_str = 'x[0] > 0.5 ?'\
                'kappa_1 : kappa_0'

            # Physical parameters
            kappa = Expression(kappa_str, kappa_0=kappa_0,
                               kappa_1=kappa_1, degree=1)
            # Define initial condition(initial temp of plate)
            T_current = interpolate(Constant(T_R), V)

            # Define variational problem
            T = TrialFunction(V)

            # two f's and L's for heat source on and off
            f_heat = Expression(
                'amp*exp(-(x[0]-px)*(x[0]-px)/width)', amp=amp, px=px, width=width, degree=1)
            f_cool = Constant(0)
            v = TestFunction(V)
            a = rho*cap*T*v*dx + dt*kappa * \
                inner(nabla_grad(v), nabla_grad(T))*dx
            L_heat = (rho*cap*T_current*v + dt*f_heat*v)*dx
            L_cool = (rho*cap*T_current*v + dt*f_cool*v)*dx

            A = assemble(a)
            b = None  # variable used for memory savings in assemble calls

            T = Function(V)
            t = dt  # initialize first time step
            print("%d Starting timestepping."%i)
            # time stepping method is BWD Euler. (theta = 1)
            while t <= t_stop:
                if t < t_heatoff:
                    b = assemble(L_heat, tensor=b)
                else:
                    b = assemble(L_cool, tensor=b)
                solve(A, T.vector(), b)

                t += dt
                T_current.assign(T)

            # now that the state variable (Temp at time t_stop) has been computed,
            # we take our point-value measurements
            QoI_samples[i] = np.array([T(xi) for xi in temp_locs_list])
        return QoI_samples

    return my_model
