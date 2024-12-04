import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """
        # Dynamics
        # TODO implement the dynamics -- R: DONE (should be correct 95%)
        f = spy.zeros(self.n_x, 1)  # n_x is 8

        # Dynamics (dx/dt, dy/dt, dpsi/dt)
        f[0] = self.x[3] * spy.cos(self.x[2]) - self.x[4] * spy.sin(self.x[2])  # Update of x
        f[1] = self.x[3] * spy.sin(self.x[2]) + self.x[4] * spy.cos(self.x[2])  # Update of y (it's just vy)
        f[2] = self.x[5]  # Update of psi (it's just dpsi)

        # Come vengono aggiornati self.x, self.u e self.p???
        # Velocity dynamics (dvx/dt, dvy/dt, ddpsi/dt)
        f[3] = 1 / self.x[7] * spy.cos(self.x[6]) * self.u[0] + self.x[5] * self.x[4]  # Update of vx
        f[4] = 1 / self.x[7] * spy.sin(self.x[6]) * self.u[0] - self.x[5] * self.x[3]  # Update of vy
        f[5] = 1 / self.sg.Iz * (-spy.sin(self.x[6]) * self.u[0] * self.sg.l_r)  # Update of dpsi

        # Control dynamics
        f[6] = self.u[1]  # Update of delta

        # Mass dynamics (not important anymore)
        f[7] = -self.sp.C_T * self.u[0]

        f = f * self.p[0]  # F: I do not know if this step is required

        '''
        print(
            f"dx/dt: {f[0]}\n\
                dy/dt: {f[1]}\n\
                dpsi/dt: {f[2]}\n\
                dvx/dt: {f[3]}\n\
                dvy/dt: {f[4]}\n\
                ddpsi/dt: {f[5]}\n\
                ddelta/dt: {f[6]}\n\
                dm/dt: {f[7]}\n\
              "
        )
        '''

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func