import dolfin as df
import numpy as np

def make_model(temp_locs_list, end_time = 1.0):
    """
    TODO: put parameters into options
    """
    # heatrod code here - including all settings
    t_stop = end_time

    # Some fixed parameter values
    amp = 50.0  # amplitude of the heat source
    px = 0.5  # location of the heat source
    width = 0.05  # width of the heat source
    T_R = 0  # initial temp of the plate
    cap = 1.5  # heat capacity
    rho = 1.5  # density

    # 'parameters' reserved for FEnICS
    df.parameters['allow_extrapolation'] = True

    # mesh properties
    nx = 50  # this is our h value
    mesh = df.IntervalMesh(nx, 0, 1)
    degree = 1
    r = 1.0  # time stepping ratios - attention to stability
    dt = r/nx

    # turn off the heat halfway through
    t_heatoff = 0.5

    def model(parameter_samples):
        """
        TODO: string-formatting that displays the models parameters.

        Returns model evaluated at `T={end_time}`.
        """.format(end_time=end_time)

        if parameter_samples.ndim == 1:
            assert len(parameter_samples) == 2
            num_samples = 1
        else:
            num_samples = parameter_samples.shape[0]
        QoI_samples = np.zeros((num_samples, len(temp_locs_list)))

        for i in range(num_samples):
            try:
                kappa_0 = parameter_samples[i, 0]
                kappa_1 = parameter_samples[i, 1]
            except IndexError:
                kappa_0 = parameter_samples[0]
                kappa_1 = parameter_samples[1]

            # define the subspace we will solve the problem in
            V = df.FunctionSpace(mesh, 'Lagrange', degree)

            # split the domain down the middle(dif therm cond)
            kappa_str = 'x[0] > 0.5 ?'\
                'kappa_1 : kappa_0'

            # Physical parameters
            kappa = df.Expression(kappa_str, kappa_0=kappa_0,
                               kappa_1=kappa_1, degree=1)
            # Define initial condition(initial temp of plate)
            T_current = df.interpolate(df.Constant(T_R), V)

            # Define variational problem
            T = df.TrialFunction(V)

            # two f's and L's for heat source on and off
            f_heat = df.Expression(
                'amp*exp(-(x[0]-px)*(x[0]-px)/width)', amp=amp, px=px, width=width, degree=1)
            f_cool = df.Constant(0)
            v = df.TestFunction(V)
            a = rho*cap*T*v*df.dx + dt*kappa * \
                df.inner(df.nabla_grad(v), df.nabla_grad(T))*df.dx
            L_heat = (rho*cap*T_current*v + dt*f_heat*v)*df.dx
            L_cool = (rho*cap*T_current*v + dt*f_cool*v)*df.dx

            A = df.assemble(a)
            b = None  # variable used for memory savings in assemble calls

            T = df.Function(V)
            t = dt  # initialize first time step
            print("%d Starting timestepping."%i)
            # time stepping method is BWD Euler. (theta = 1)
            while t <= t_stop:
                if t < t_heatoff:
                    b = df.assemble(L_heat, tensor=b)
                else:
                    b = df.assemble(L_cool, tensor=b)
                df.solve(A, T.vector(), b)

                t += dt
                T_current.assign(T)

            # now that the state variable (Temp at time t_stop) has been computed,
            # we take our point-value measurements
            QoI_samples[i,:] = np.array([T(xi) for xi in temp_locs_list])
        # if QoI_samples.shape[0] == 1:
        #     QoI_samples = QoI_samples.ravel()[0]
        return QoI_samples

    return model


