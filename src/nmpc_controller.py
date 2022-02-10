import casadi as ca
import numpy as np
import math

class NMPCController:
    def __init__(self, init_pos, min_vx, max_vx, min_vy, max_vy, min_omega, max_omega,
                T=0.02, N=30, Q=np.diag([50.0, 50.0, 10.0]), R=np.diag([1.0, 1.0, 1.0])):
        self.T = T          # time step
        self.N = N          # horizon length

        self.Q = Q          # Weight matrix for states
        self.R = R          # Weight matrix for controls

        # Constraints
        self.min_vx = min_vx
        self.max_vx = max_vx

        self.min_vy = min_vy
        self.max_vy = max_vy

        self.min_omega = min_omega
        self.max_omega = max_omega

        self.max_dvx = 0.8
        self.max_dvy = 0.8
        self.max_domega = math.pi/6
    
        # The history states and controls
        self.next_states = np.ones((self.N+1, 3))*init_pos
        self.u0 = np.zeros((self.N, 3))

        self.setup_controller()
    
    def setup_controller(self):
        self.opti = ca.Opti()

        # state variable: position and velocity
        self.opt_states = self.opti.variable(self.N+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]

        # the velocity
        self.opt_controls = self.opti.variable(self.N, 3)
        vx = self.opt_controls[0]
        vy = self.opt_controls[1]
        omega = self.opt_controls[2]

        # # the first derivative velocity
        # self.opt_dcontrols = self.opti.variable(self.N, 3)
        # dvx = self.opt_dcontrols[0]
        # dvy = self.opt_dcontrols[1]
        # domega = self.opt_dcontrols[2]

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            ca.cos(x_[2])*u_[0] - ca.sin(x_[2])*u_[1],  # dx
            ca.sin(x_[2])*u_[0] + ca.cos(x_[2])*u_[1],  # dy
            u_[2],                                      # dtheta
        ])

        # parameters, these parameters are the reference trajectories of the pose and inputs
        self.opt_u_ref = self.opti.parameter(self.N, 3)
        self.opt_x_ref = self.opti.parameter(self.N+1, 3)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T*self.T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)
        
        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i+1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                        + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # constraint about change of velocity
        for i in range(self.N-1):
            dvel = (self.opt_controls[i+1,:] - self.opt_controls[i,:])/self.T
            self.opti.subject_to(self.opti.bounded(-self.max_dvx, dvel[0], self.max_dvx))
            self.opti.subject_to(self.opti.bounded(-self.max_dvy, dvel[1], self.max_dvy))
            self.opti.subject_to(self.opti.bounded(-self.max_domega, dvel[2], self.max_domega))

        # boundary and control conditions
        self.opti.subject_to(self.opti.bounded(self.min_vx, vx, self.max_vx))
        self.opti.subject_to(self.opti.bounded(self.min_vy, vy, self.max_vy))
        self.opti.subject_to(self.opti.bounded(self.min_omega, omega, self.max_omega))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}

        self.opti.solver('ipopt', opts_setting)
    
    def solve(self, next_trajectories, next_controls):
        ## set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        
        ## provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        ## solve the problem
        sol = self.opti.solve()
        
        ## obtain the control input
        self.u0 = sol.value(self.opt_controls)
        self.next_states = sol.value(self.opt_states)
        return self.u0[0,:]
