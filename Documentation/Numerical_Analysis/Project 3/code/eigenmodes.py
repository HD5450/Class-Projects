# Here is Number 2

import matplotlib as mpl
import numpy as np
import scipy.linalg as la

mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["figure.figsize"] = (10, 6)


def compute_eigen_values(A, B=None):
    """Return sorted eigenvector/eigenvalue pairs.
    e.g. for a given system linalg.eig will return eingenvalues as:
    (array([ 0. +89.4j,  0. -89.4j,  0. +89.4j,  0. -89.4j,  0.+983.2j,
             0.-983.2j,  0. +40.7j,  0. -40.7j])
    This function will sort this eigenvalues as:
    (array([ 0. +40.7j,  0. +89.4j,  0. +89.4j,  0.+983.2j,  0. -40.7j,
             0. -89.4j,  0. -89.4j,  0.-983.2j])

    Correspondent eigenvectors will follow the same order.
    Note: Works fine for moderately sized models. Does not leverage the
    full set of constraints to optimize the solution.
    Parameters
    ----------
    A: array
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    B: float or str
        Right-hand side matrix in a generalized eigenvalue problem.
        Default is None, identity matrix is assumed.
    Returns
    -------
    evalues: array
        Sorted eigenvalues
    evectors: array
        Sorted eigenvalues
    Examples
    --------
    >>> L = np.array([[2, -1, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> lam, P = compute_eigen_values(L)
    >>> lam
    array([  0.56+0.j,   2.63+0.j,  10.81+0.j])
    """
    if B is None:
        evalues, evectors = la.eig(A)
    else:
        evalues, evectors = la.eig(A, B)

    if all(eigs == 0 for eigs in evalues.imag):
        if all(eigs > 0 for eigs in evalues.real):
            idxp = evalues.real.argsort()  # positive in increasing order
            idxn = np.array([], dtype=int)
        else:
            # positive in increasing order
            idxp = evalues.real.argsort()[int(len(evalues) / 2):]
            # negative in decreasing order
            idxn = evalues.real.argsort()[int(len(evalues) / 2) - 1:: -1]

    else:
        # positive in increasing order
        idxp = evalues.imag.argsort()[int(len(evalues) / 2):]
        # negative in decreasing order
        idxn = evalues.imag.argsort()[int(len(evalues) / 2) - 1:: -1]

    idx = np.hstack([idxp, idxn])

    return evalues[idx], evectors[:, idx]


def normalize(X, Y):
    """
    Return normalized left eigenvectors.
    This function is used to normalize vectors of the matrix
    Y with respect to X so that Y.T @ X = I (identity).
    This is used to normalize the matrix with the left eigenvectors.
    Parameters
    ----------
    X: array
        A complex or real matrix
    Y: array
        A complex or real matrix to be normalized
    Returns
    -------
    Yn: array
        Normalized matrix
    Examples
    --------
    >>>
    >>> X = np.array([[ 0.84+0.j  ,  0.14-0.j  ,  0.84-0.j  ,  0.14+0.j  ],
    ...               [ 0.01-0.3j ,  0.00+0.15j,  0.01+0.3j ,  0.00-0.15j],
    ...               [-0.09+0.42j, -0.01+0.65j, -0.09-0.42j, -0.01-0.65j],
    ...               [ 0.15+0.04j, -0.74+0.j  ,  0.15-0.04j, -0.74-0.j  ]])
    >>> Y = np.array([[-0.03-0.41j,  0.04+0.1j , -0.03+0.41j,  0.04-0.1j ],
    ...            [ 0.88+0.j  ,  0.68+0.j  ,  0.88-0.j  ,  0.68-0.j  ],
    ...            [-0.21-0.j  ,  0.47+0.05j, -0.21+0.j  ,  0.47-0.05j],
    ...            [ 0.00-0.08j,  0.05-0.54j,  0.00+0.08j,  0.05+0.54j]])
    >>> Yn = normalize(X, Y)
    >>> Yn
    array([[ 0.58-0.05j,  0.12-0.06j,  0.58+0.05j,  0.12+0.06j],
           [ 0.01+1.24j, -0.07-0.82j,  0.01-1.24j, -0.07+0.82j],
           [-0.  -0.3j ,  0.01-0.57j, -0.  +0.3j ,  0.01+0.57j],
           [ 0.11-0.j  , -0.66-0.01j,  0.11+0.j  , -0.66+0.01j]])
    """
    Yn = np.zeros_like(X)
    YTX = Y.T @ X  # normalize y so that Y.T @ X will return I
    factors = [1 / a for a in np.diag(YTX)]
    # multiply each column in y by a factor in 'factors'
    for col in enumerate(Y.T):
        Yn[col[0]] = col[1] * factors[col[0]]
    Yn = Yn.T

    return Yn


def undamped_modes_system(M, K):
    r"""Return eigensolution of multiple DOF system.
    Returns the natural frequencies (w),
    eigenvectors (P), mode shapes (S) and the modal transformation
    matrix S for an undamped system.
    See Writeup for explanation of the underlying math.
    Parameters
    ----------
    M: float array
        Mass matrix
    K: float array
        Stiffness matrix
    Returns
    -------
    w: float array
        The natural frequencies of the system
    P: float array
        The eigenvectors of the system.
    S: float array
        The mass-normalized mode shapes of the system.
    Sinv: float array
        The modal transformation matrix S^-1(takes x -> r(modal coordinates))
    Notes
    -----
    Given :math:`M\ddot{x}(t)+Kx(t)=0`, with mode shapes :math:`u`, the matrix
    of mode shapes :math:`S=[u_1 u_1 \ldots]` can be created. If the modal
    coordinates are the vector :math:`r(t)`. The modal transformation separates
    space and time from :math:`x(t)` such that :math:`x(t)=S r(t)`.
    Substituting into the governing equation:
    :math:`MS\ddot{r}(t)+KSr(t)=0`
    Premultiplying by :math:`S^T`
    :math:`S^TMS\ddot{r}(t)+S^TKSr(t)=0`

    The matrices :math:`S^TMS` and :math:`S^TKS` will be diagonalized by this
    process (:math:`u_i` are the eigenvectors of :math:`M^{-1}K`).
    If scaled properly (mass normalized so :math:`u_i^TMu_i=1`) then
    :math:`S^TMS=I` and :math:`S^TKS=\Omega^2` where :math:`\Omega^2` is a
    diagonal matrix of the natural frequencies squared in radians per second.

    Further, inverses are unstable so the better way to solve linear equations is with
    Gauss elimination.
    :math:`AB=C` given known :math:`A` and :math:`C`
    is solved using `la.solve(A, C, assume_a='pos')`.
    :math:`BA=C` given known :math:`A` and :math:`C` is solved by first
    transposing the equation to :math:`A^TB^T=C^T`, then solving for
    :math:`C^T`. The resulting command is
    `la.solve(A.T, C.T, assume_a='pos').T`
    Examples
    --------
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> w, P, S, Sinv = modes_system_undamped(M, K)
    >>> w
    array([0.45, 1.25, 1.8 ])
    >>> S
    array([[ 0.16, -0.37, -0.3 ],
           [ 0.3 , -0.16,  0.37],
           [ 0.37,  0.3 , -0.16]])
    """
    L = la.cholesky(M)
    lam, P = compute_eigen_values(la.solve(L, la.solve(L, K, assume_a='pos').T,
                                           assume_a='pos').T)
    w = np.real(np.sqrt(lam))
    S = la.solve(L, P, assume_a="pos")
    Sinv = la.solve(L.T, P, assume_a="pos").T

    return w, P, S, Sinv


def damped_modes_system(M, K, C=None):
    """Natural frequencies, damping ratios, and mode shapes of the system.
    This function will return the natural frequencies (wn), the
    damped natural frequencies (wd), the damping ratios (zeta),
    the right eigenvectors (X) and the left eigenvectors (Y) for a
    system defined by M, K and C.
    If the dampind matrix 'C' is none or if the damping is proportional,
    wd and zeta will be none and X and Y will be equal.
    Parameters
    ----------
    M: array
        Mass matrix
    K: array
        Stiffness matrix
    C: array
        Damping matrix
    Returns
    -------
    wn: array
        The natural frequencies of the system
    wd: array
        The damped natural frequencies of the system
    zeta: array
        The damping ratios
    X: array
        The right eigenvectors
    Y: array
        The left eigenvectors
    Examples
    --------
    >>> M = np.array([[1, 0],
    ...               [0, 1]])
    >>> K = np.array([[2, -1],
    ...               [-1, 6]])
    >>> C = np.array([[0.3, -0.02],
    ...               [-0.02, 0.1]])
    >>> wn, wd, zeta, X, Y = modes_system(M, K, C)
    Damping is non-proportional, eigenvectors are complex.
    >>> wn
    array([1.33, 2.5 , 1.33, 2.5 ])
    >>> wd
    array([1.32, 2.5 , 1.32, 2.5 ])
    >>> zeta
    array([0.11, 0.02, 0.11, 0.02])
    >>> X
    array([[-0.06-0.58j, -0.01+0.08j, -0.06+0.58j, -0.01-0.08j],
           [-0.  -0.14j, -0.01-0.36j, -0.  +0.14j, -0.01+0.36j],
           [ 0.78+0.j  , -0.21-0.03j,  0.78-0.j  , -0.21+0.03j],
           [ 0.18+0.01j,  0.9 +0.j  ,  0.18-0.01j,  0.9 -0.j  ]])
    >>> Y
    array([[ 0.02+0.82j,  0.01-0.31j,  0.02-0.82j,  0.01+0.31j],
           [-0.05+0.18j,  0.01+1.31j, -0.05-0.18j,  0.01-1.31j],
           [ 0.61+0.06j, -0.12-0.02j,  0.61-0.06j, -0.12+0.02j],
           [ 0.14+0.03j,  0.53+0.j  ,  0.14-0.03j,  0.53-0.j  ]])
    >>> C = 0.2*K # with proportional damping
    >>> wn, wd, zeta, X, Y = modes_system(M, K, C)
    Damping is proportional or zero, eigenvectors are real
    >>> X
    array([[-0.97,  0.23],
           [-0.23, -0.97]])
    """

    n = len(M)

    Z = np.zeros((n, n))
    I = np.eye(n)

    if (
                        C is None or
                    np.all(C == 0) or
                    la.norm(  # check if C has only zero entries
                                la.solve(M, C, assume_a="pos") @ K - \
                                    la.solve(M, K, assume_a="pos") @ C, 2
                    ) <
                        1e-8 * la.norm(la.solve(M, K, assume_a="pos") @ C, 2)
    ):
        w, P, S, Sinv = modes_system_undamped(M, K)
        wn = w
        wd = w
        # zeta = None
        zeta = np.diag(S.T @ C @ S) / 2 / wn
        wd = wn * np.sqrt(1 - zeta ** 2)
        X = P
        Y = P
        print("Damping is proportional or zero, eigenvectors are real")
        return wn, wd, zeta, X, Y

    Z = np.zeros((n, n))
    I = np.eye(n)

    # creates the state space matrix
    A = np.vstack(
        [
            np.hstack([Z, I]),
            np.hstack(
                [-la.solve(M, K, assume_a="pos"),
                 - la.solve(M, C, assume_a="pos")]
            ),
        ]
    )

    w, X = compute_eigen_values(A)
    _, Y = compute_eigen_values(A.T)

    wd = abs(np.imag(w))
    wn = np.absolute(w)
    zeta = -np.real(w) / np.absolute(w)

    Y = normalize(X, Y)

    print("Damping is non-proportional, eigenvectors are complex.")

    return wn, wd, zeta, X, Y
