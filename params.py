class Params:

    def __init__(self):
        # repressilator parameters
        # self.alpha = 216
        # self.alpha0 = 0.001 * self.alpha
        # self.n = 2
        # self.beta = 5

        self.alpha = 5                      # min ** (-1)
        self.alpha0 = 0.001 * self.alpha    # min ** (-1)
        self.beta = 1                       # min ** (-1)
        self.Kd = 10                        # nM
        self.delta_p = 0.1                  # min ** (-1)
        self.delta_m = 0.1                  # min ** (-1)
        self.n = 2

        self.kappa = 0.2
        self.kS0 = 1
        self.kS1 = 0.01
        self.kSe = 0.01
        self.eta = 2

        # diffusion rate
        self.D1 = 0.5

        # environment
        self.size = 10 # size of the space equals[size x size]
        self.density = 0.8

        # simulation parameters
        self.t_end = 10000
        self.dt = 0.1
        self.h = 0.5 # Grid size: in micro meters - E coli size 1 um x 2 um(volume = 1 um ^ 3)
        self.h2 = self.h ** 2