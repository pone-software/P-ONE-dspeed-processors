from __future__ import annotations

import numpy as np
from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs
from numba import guvectorize, njit


@njit
def merge_close(p, s, a, b, greater_than_min, allowed, n=5):
    solns = np.where(p)[0]

    if len(solns) == 0:
        return p, s
    groups = []
    current_group = [solns[0]]

    for i in range(1, len(solns)):
        # Check if current index is within n of the first index in current group
        if solns[i] - current_group[0] <= n:
            current_group.append(solns[i])
        else:
            # Finalize current group and start new one
            groups.append((min(current_group), max(current_group)))
            current_group = [solns[i]]

    # Don't forget the last group
    if len(current_group) > 1:
        groups.append((min(current_group), max(current_group)))

    if len(groups) > 0:
        for entry in groups:
            n, n2 = entry
            tot = np.sum(s[n : n2 + 1])
            reses = np.zeros(n2 - n + 1, dtype=a.dtype)
            for i, j in enumerate(range(n, n2 + 1)):
                test = s.copy()
                test[n : n2 + 1] = 0
                test[j] = tot
                reses[i] = np.sum(
                    (
                        b[greater_than_min]
                        - np.dot(a[greater_than_min, :][:, allowed], test[allowed])
                    )
                    ** 2
                )
            s[n : n2 + 1] = 0
            s[np.argmin(reses) + n] = tot
            p[n : n2 + 1] = False
            p[np.argmin(reses) + n] = True

    return p, s


@guvectorize(
    [
        "void(float32[:,::1], float32[::1], float32[:,::1], float32, float32, boolean, float32, float32, float32[::1])",
        "void(float64[:,::1], float64[::1], float64[:,::1], float64, float32, boolean, float32,float32, float64[::1])",
    ],
    "(m,n),(m),(n,n),(),(),(),(),(),(n)",
    # nopython=True,
    forceobj=True,
    **nb_kwargs,
)
def optimize_nnls_mod(
    a: np.ndarray,
    b: np.ndarray,
    ata: np.ndarray,
    maxiter: int,
    tol: float,
    allow_singularity: bool,
    min_value: float,
    batch_size: int,
    x: np.ndarray,
    # verbose=False
) -> None:
    """Solve ``argmin_x || ax - b ||_2`` for ``x>=0``.
    Based on :func:`scipy.optimize.nnls` implementation. Which in turn is based on
    the algorithm in Bro, R. and De Jong, S. (1997), A fast non-negativity-constrained least squares algorithm. J. Chemometrics, 11: 393-401

    Parameters
    ----------
    a : (m, n) ndarray
        Coefficient matrix
    b : (m,) ndarray, float
        Right-hand side vector.
    ata: ndarray
        a.T @ a
    maxiter: int
        Maximum number of iterations.
    tol: float
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(a.T @ (a x - b)`` entries. Increasing this
        value relaxes the solution constraints.
    allow_singularity: bool
        If matrix is not solvable (e.g. because of non full rank caused by
        float precision), no error is raised but all elements of the
        solution vector are set NaN
    min_value: float
        threshold value.
    batch_size: int
    x : ndarray
        Solution vector.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "nnls_solution": {
            "function": "optimize_nnls",
            "module": "dspeed.processors",
            "args": ["db.coefficient_matrix",
                "wf_blsub",
                "1000", "1e-6", "True"
                "nnls_solution"],
        }
    """

    def numba_ix(arr: np.array, rows: np.array, cols: np.array) -> np.array:
        """
        Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
        from https://github.com/numba/numba/issues/5894#issuecomment-974701551
        :param arr: 2D array to be indexed
        :param rows: Row indices
        :param cols: Column indices
        :return: 2D array with the given rows and columns of the input array
        """
        one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
        for i, r in enumerate(rows):
            start = i * len(cols)
            one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

        arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
        slice_1d = np.take(arr_1d, one_d_index)
        return slice_1d.reshape((len(rows), len(cols)))

    def is_singular(matrix):
        """
        Returns True if matrix det = 0 i.e. matrix is singular.
        """
        det = np.linalg.det(matrix)
        return abs(det) < np.finfo(np.float64).eps

    m, n = a.shape

    if n != len(x):
        msg = "n dimension of coefficient axis doesn't match solution vector length."
        raise DSPFatal(msg)
    if m != len(b):
        msg = "m dimension of coefficient axis doesn't match right-hand vector length."
        raise DSPFatal(msg)

    # ata = a.T @ a
    atb = b @ a  # Result is 1D - let NumPy figure it out

    template_max = np.amax(b)
    greater_than_min = b > (0.1 * template_max)
    ids = np.where(greater_than_min)[0]
    for i in ids:
        greater_than_min[i - 2 : i + 2] = True

    allowed = np.zeros(a.shape[1], dtype=bool)
    for i in range(a.shape[1]):
        if np.any(a[greater_than_min, i]):
            allowed[i] = True

    # if verbose: print(np.where(allowed)[0])

    # Initialize vars
    x[:] = np.zeros(n, dtype=a.dtype)
    s = np.zeros(n, dtype=a.dtype)
    # Inactive constraint switches
    p = np.zeros(n, dtype=np.bool_)
    pidx = np.arange(0, len(p), 1, dtype=np.int32)

    # Projected residual
    w = atb.copy().astype(a.dtype)  # x=0. Skip (-ata @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0
    tried = np.zeros(n, dtype=np.bool_)
    while (not p.all()) and (not tried.all()) and (w[~p * ~tried * allowed] > tol).any():
        # Get the "most" active coeff index and move to inactive set
        k = np.argmax(w * (~p * ~tried * allowed))
        # if verbose : print("k:",k)
        if batch_size != 0:
            lo = max(0, k - int(batch_size / 2))
            hi = min(len(p), k + int(batch_size / 2))
            res = b[greater_than_min] - a[greater_than_min, :][:, lo:hi] @ x[lo:hi]
            gradient = a[greater_than_min, :][:, lo:hi].T @ res
            sort_grad = np.argsort(gradient)[::-1]
            sort_grad = np.delete(sort_grad, np.where(sort_grad == k)[0])
            order = [k, *np.arange(lo, hi, 1)[sort_grad]]
        else:
            order = [k]

        for i in order:

            p[i] = True
            tried[i] = True
            # if verbose: print(i, np.where(p)[0])
            # Iteration solution
            s[:] = 0.0
            mat = numba_ix(ata, pidx[p], pidx[p])

            # check if matrix has full rank before solving
            if is_singular(mat) and allow_singularity:
                x[:] = np.nan
                return

            s[p] = np.linalg.solve(mat, atb[p])

            # if verbose: print(np.where(p)[0], s[p])

            # Inner loop
            while (iter < maxiter) and len(s[p]) > 0 and (s[p].min() <= min_value):
                p, s = merge_close(p, s, a, b, greater_than_min, allowed)
                if s[p].min() >= min_value:
                    break
                iter += 1
                inds = p * (s <= min_value)
                alpha = (x[inds] / (x[inds] - s[inds])).min()
                # if alpha <0: alpha=0
                x *= 1 - alpha
                x += alpha * s
                # if verbose : print(x[p])
                p[x <= min_value] = False

                mat = numba_ix(ata, pidx[p], pidx[p])
                if is_singular(mat) and allow_singularity:
                    x[:] = np.nan
                    return

                s[p] = np.linalg.solve(mat, atb[p])
                s[~p] = 0

                # if verbose: print(np.where(p)[0], s[p])

            # if verbose: print("pre-final", np.where(p)[0], s[p])
            p, new_s = merge_close(p, s, a, b, greater_than_min, allowed)
            if (new_s != s).all():
                mat = numba_ix(ata, pidx[p], pidx[p])
                s[p] = np.linalg.solve(mat, atb[p])
            # if verbose: print("final", np.where(p)[0], s[p])
            x[allowed] = s[allowed]
            w[allowed] = atb[allowed] - ata[allowed, :][:, allowed] @ x[allowed]

        if iter == maxiter:
            return

        # if verbose: print((not p.all()),  (not tried.all()), (w[~p * ~tried * allowed] > tol).any(), np.where(~p * ~tried * allowed)[0], w[~p * ~tried * allowed])
    # if verbose: print(np.where(x>0)[0], x[x>0])
