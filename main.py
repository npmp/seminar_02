from __future__ import division

from random import randint, random

from numpy import matrix, reshape
from numpy.ma import ceil, zeros, array
from numpy.matlib import rand

from params import Params


class Repressilator:

    def run(self):
        # ali rob predstavlja konec prostora ali so meje neskončne?
        periodic_bounds = 1
        # nalaganje shranjene konfiguracije?
        load_conf = 0
        # shranjevanje končne konfiguracije?
        save_conf = 0
        # fiksni robovi ali spremenljivi
        borderfixed = 0
        # snemanje videa - časovno potratno
        movie_on = 0

        # nalaganje vrednosti parametrov
        p = Params()
        alpha = p.alpha
        alpha0 = p.alpha0
        Kd = p.Kd
        delta_m = p.delta_m
        delta_p = p.delta_p
        n = p.n
        beta = p.beta
        kappa = p.kappa
        kS0 = p.kS0
        kS1 = p.kS1
        kSe = p.kSe
        D1 = p.D1
        eta = p.eta

        size = p.size
        density = p.density
        n_cells = int(ceil(density * size ** 2))

        t_end = p.t_end
        dt = p.dt
        h = p.h
        h2 = p.h2

        # S = zeros(size, size) # Initialise species S
        S_e = rand((size, size))
        S_i = zeros((size, size))
        A = zeros((size, size))
        B = zeros((size, size))
        C = zeros((size, size))
        mA = zeros((size, size))
        mB = zeros((size, size))
        mC = zeros((size, size))

        CELLS = zeros((size, size))
        cell_idx = zeros(n_cells)

        j = 0
        for i in range (0, n_cells):
            j = i
            xi = randint(0, size - 1)
            yi = randint(0, size - 1)
            while CELLS[xi][yi] == 1:
                xi = randint(0, size - 1)
                yi = randint(0, size - 1)

            CELLS[xi][yi] = 1
            cell_idx[j] = xi * size + yi

        cell_idx = [int(x) for x in sorted(cell_idx)]

        # cell_idx = ceil(size ** 2 * rand(1, n_cells))
        # CELLS = zeros(size, size)
        # CELLS[cells_idx - 1 = 1

        first_idx = cell_idx[0]

        A = A.reshape(size ** 2, 1)
        mA = mA.reshape(size ** 2, 1)
        B = B.reshape(size ** 2, 1)
        mB = mB.reshape(size ** 2, 1)
        C = C.reshape(size ** 2, 1)
        mC = mC.reshape(size ** 2, 1)
        A[cell_idx] = 100 * rand(len(cell_idx), 1)
        mA[cell_idx] = 100 * rand(len(cell_idx), 1)
        B[cell_idx] = 100 * rand(len(cell_idx), 1)
        mB[cell_idx] = 100 * rand(len(cell_idx), 1)
        C[cell_idx] = 100 * rand(len(cell_idx), 1)
        mC[cell_idx] = 100 * rand(len(cell_idx), 1)

        A_series = zeros((1, int(t_end / dt)))
        S_e_series = zeros((1, int(t_end / dt)))
        A_full = zeros((int(t_end / dt), n_cells))

        A_series[0] = A[first_idx]
        S_e_series[0, 0] = S_e[first_idx][0, 0]
        A_full[0, :] = A[cell_idx].ravel()

        t = 0
        k = 0
        step = 0

        D2S_e = None

        while t <= t_end:

            if (periodic_bounds):
                cols = S_e[0].size - 1
                rows = S_e[:][0].size - 1

                S_e_xx = D1 * (S_e[:][[cols] + list(range(0, cols))] + S_e[:][list(range(1, cols + 1)) + [0]] - 2 * S_e) / h2
                S_e_yy = D1 * (S_e[[rows] + list(range(0, rows))][:] + S_e[list(range(1, rows + 1)) + [0]][:] - 2 * S_e) / h2
            else:
                # Create padded matrix to incorporate Neumann boundary conditions
                SS_e = array(array(0, S_e[1][:], 0), array(S_e[:][1], S_e, S_e[:][-2]), array(0, S_e[-2][:], 0))

                # Calculate diffusion part of the equations
                S_e_xx = D1 * (SS_e[1:-2][0:-3] + SS_e[2:-2][3:-1] -2 * S_e) / h2
                S_e_yy = D1 * (SS_e[1:-3][1:-2] + SS_e[2:-1][1:-2] -2 * S_e) / h2

            D2S_e = S_e_xx + S_e_yy

            # Calculate dx / dt
            dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e = repressilator_S_ODE(CELLS, mA, mB, mC, A, B, C, S_i, S_e, alpha,
                                                                        alpha0, Kd, beta, delta_m, delta_p, n, kS0, kS1,
                                                                        kSe, kappa, eta, size)

            dS_e = dS_e + D2S_e

            if (borderfixed == 1):
                # leave border as distortion centers
                width = len(dS_e)
                dS_e[0:width][0, width] = 0
                dS_e[0, width][0:width] = 0

            # reshape back to original values (size, size)
            # TODO: optimise this
            A = A.reshape(size, size)
            mA = mA.reshape(size, size)
            B = B.reshape(size, size)
            mB = mB.reshape(size, size)
            C = C.reshape(size, size)
            mC = mC.reshape(size, size)

            mA = mA + dt * dmA
            mB = mB + dt * dmB
            mC = mC + dt * dmC
            A = A + dt * dA
            B = B + dt * dB
            C = C + dt * dC
            S_i = S_i + dt * dS_i
            S_e = S_e + dt * dS_e

            t = t + dt
            step = step + 1

            A_series[0][step - 1] = A[first_idx - 1][first_idx - 1]
            S_e_series[0][step - 1] = S_e[first_idx - 1][first_idx - 1]
            A_full[step - 1][:] = A.flatten()[cell_idx]

            print(step)

def repressilator_S_ODE(CELLS, mA, mB, mC, A, B, C, S_i, S_e, alpha, alpha0, Kd, beta, delta_m, delta_p, n, kS0, kS1,
                        kSe, kappa, eta, size):

    dmA = CELLS * reshape(alpha / (1 + (C / Kd) ** n) + alpha0 - delta_m * mA, (size, size))
    dmB = CELLS * reshape(alpha / (1 + (A / Kd) ** n) + alpha0 - delta_m * mB, (size, size))
    dmC = CELLS * (reshape(alpha / (1 + (B / Kd) ** n) + alpha0 - delta_m * mC, (size, size)) + (kappa * S_i) / (1 + S_i))

    dA = CELLS * reshape(beta * mA - delta_p * A, (size, size))
    dB = CELLS * reshape(beta * mB - delta_p * B, (size, size))
    dC = CELLS * reshape(beta * mC - delta_p * C, (size, size))

    # reshape back to original values (size, size)
    # TODO: only do this once
    A = A.reshape(size, size)
    dS_i = CELLS * (- kS0 * S_i + kS1 * A - eta * (S_i - S_e))
    dS_e = - kSe * S_e + CELLS * (eta * (S_i - S_e))

    return dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e

if __name__ == "__main__":
    rep = Repressilator()
    rep.run()