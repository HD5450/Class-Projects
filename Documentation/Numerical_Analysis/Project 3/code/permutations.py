import numpy as np
import itertools as it
import eigenmodes as em


if __name__ == "__main__":

    m1 = 0;
    m2 = 0;

    k1 = 0;
    k2 = 0;
    k3 = 0;

    c1 = 0;

    target = -2 + 2j

    M3 = list(np.arange(0.1, 0.7, 0.1))
    M4 = list(np.arange(0.0, 0.05, 0.01))

    C2 = list(np.arange(0.15, 0.60, 0.1))
    C3 = list(np.arange(0.015, 0.03, 0.01))

    K4 = list(np.arange(0.01, 0.2, 0.01))
    K5 = list(np.arange(0.1, 1, 0.1))

    l = [M3, M4, C2, C3, K4, K5]

    combinations = list(it.product(*l))

    collected_params = []

    for _, permutation in enumerate(combinations):

        m3 = permutation[0]
        m4 = permutation[1]

        c2 = permutation[2]
        c3 = permutation[3]

        k4 = permutation[4]
        k5 = permutation[5]

        M = np.array([[m3, 0],
                      [0, m4]])

        K = np.array([[k3 + k4 + k5, -k4],
                      [-k4, k4]])

        C = np.array([[c2 + c3, -c3],
                      [-c3, c3]])

        try:
            wn, wd, zeta, X, Y = em.modes_system(M, K, C)
        except:
            continue

        if target.imag - wd[0] < 1e-5:
            params = {
                "m3": m3,
                "m4": m4,
                "c2": c2,
                "c3": c3,
                "k4": k4,
                "k5": k5,
                "wd": wd,
                "wn": wn
            }

            collected_params.append(params)

        assert (len(collected_params) > 0)

        reduced_search_params = [item for item in collected_params
                                 if item['wd'][1] - 3.0 <= 0.000001
                                 ]

        print(reduced_search_params[0])