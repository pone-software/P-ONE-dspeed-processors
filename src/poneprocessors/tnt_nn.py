"""
Non-negative Least Squares feasible solution.

Derived from Erich Frahm and Joseph Myre's Matlab implementation
available at https://github.com/ProfMyre/tnt-nn.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as la
from dspeed.utils import nb_kwargs
from numba import guvectorize, njit

from .nnls_mod import merge_close


@njit
def chol(a: np.ndarray, lower: bool = False) -> tuple[np.ndarray, int]:
    """Cholesky factorization inspired by Matlab's chol function

    Uses LAPACK's POTRF subroutine to compute the factorization.

    Parameters
    ----------
    A : (M, M) ndarray
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper- or lower-triangular Cholesky
        factorization.  Default is upper-triangular.

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of a.
    info : int
        Status information on the POTRF call:
            * -i < 0 : the i-th argument had an illegal value.
            *    = 0 : successful execution.
            *  i > 0 : the leading minor of order i is not positive definite,
                         and the factorization could not be completed.
    """
    a1 = np.atleast_2d(np.asarray_chkfinite(a))

    # Dimension check
    if a1.ndim != 2:
        msg = f"Input array needs to be 2D but received a {a1.ndim}d-array."
        raise ValueError(msg)
    # Squareness check
    if a1.shape[0] != a1.shape[1]:
        msg = f"Input array is expected to be square but has the shape: {a1.shape}."
        raise ValueError(msg)

    # Quick return for square empty array
    if a1.size == 0:
        msg = "Empty input array."
        raise ValueError(msg)

    try:
        c = np.linalg.cholesky(a1) if lower else np.linalg.cholesky(a1).T
        info = 0
    except Exception:
        c = np.zeros_like(a1)
        info = 1  # Simplified error code

    return c, info


@njit
def pcgnr(A: np.ndarray, b: np.ndarray, R: np.ndarray, lbda=0) -> tuple[np.ndarray, int]:
    """Left-Preconditioned CGNR.

    Solves the normal equations A^T A x = A^T b using a left-preconditioned
    conjugate gradient method. See Algorithm 9.7 from "Iterative Methods
    for Sparse Linear Systems" (2nd Ed.), by Yousef Saad.

    Parameters
    ----------
    A : (m, n) ndarray
        Left-hand side matrix
    b : (m,) ndarray
        Right-hand side vector
    R : (n, n) ndarray
        Left-preconditioner matrix

    Returns
    -------
    x : (n,) ndarray
        Solution of the normal equations system
    k : int
        Number of iterations
    """
    # need to modify this to add merges
    n = A.shape[1]
    x = np.zeros(n, dtype=A.dtype)
    r = b.copy()
    r_hat = A.T @ r - lbda
    y = np.linalg.solve(R, r_hat)
    z = np.linalg.solve(R, y)
    p = z.copy()

    gamma = np.dot(z, r_hat)
    prev_rr = -1

    for k in range(1, n + 1):
        niters = k
        w = A @ p
        ww = np.dot(w, w)

        if ww == 0:
            break

        alpha = gamma / ww
        x_prev = x.copy()
        x += alpha * p
        r = b - A @ x
        r_hat = A.T @ r

        # Enforce continuous improvement in the score.
        rr = np.dot(r_hat, r_hat)
        if 0 <= prev_rr <= rr:
            x = x_prev
            break
        prev_rr = rr

        y = np.linalg.solve(R, r_hat)
        z = np.linalg.solve(R, y)
        gamma_new = np.dot(z, r_hat)
        beta = gamma_new / gamma
        p = z + beta * p
        gamma = gamma_new

        if gamma == 0:
            break

    _, x = merge_close(
        np.where(x != 0)[0],
        x,
        A,
        b,
        np.zeros(len(b), dtype=np.bool_) + 1,
        np.zeros(len(x), dtype=np.bool_) + 1,
    )

    return x, niters


@njit
def dlartg(f, g):
    """
    Numba-compatible version of LAPACK dlartg
    Generate a plane rotation so that:
    [  c  s ] [ f ]   [ r ]
    [ -s  c ] [ g ] = [ 0 ]

    Returns: (c, s, r)
    where c = cos(theta), s = sin(theta), r = sqrt(f^2 + g^2)
    """
    if g == 0.0:
        c = 1.0
        s = 0.0
        r = abs(f)
    elif f == 0.0:
        c = 0.0
        s = 1.0 if g > 0 else -1.0
        r = abs(g)
    # Use the more stable algorithm
    elif abs(f) > abs(g):
        t = g / f
        u = np.sqrt(1.0 + t * t)
        if f < 0:
            u = -u
        c = 1.0 / u
        s = t * c
        r = f * u
    else:
        t = f / g
        u = np.sqrt(1.0 + t * t)
        if g < 0:
            u = -u
        s = 1.0 / u
        c = t * s
        r = g * u

    return c, s, r


@njit
def _cholesky_delete(R: np.ndarray, BB: np.ndarray, deletion_set: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    num_deletions = len(deletion_set)

    speed_fudge_factor = 0.001
    if num_deletions > speed_fudge_factor * n:
        R, p = chol(BB)
        if p > 0:
            # This should never happen because we have already added
            # a sufficiently large "epsilon" to AA to do the
            # nonnegativity tests required to create the deleted_set.
            msg = "Could not compute Cholesky factorization"
            raise ValueError(msg)
    else:
        for i in range(num_deletions):
            j = deletion_set[i]
            # This function is just a stripped version of Matlab's qrdelete.
            # http://pmtksupport.googlecode.com/svn/trunk/lars/larsen.m

            m, n = R.shape
            new_R = np.empty((m, n - 1), dtype=R.dtype)

            # Copy columns before j
            new_R[:, :j] = R[:, :j]
            # Copy columns after j
            new_R[:, j:] = R[:, j + 1 :]
            R = new_R

            for k in range(j, n):

                p = np.array([k, k + 1])
                x = np.asarray_chkfinite(R[p, k])

                if x.shape != (2,):
                    msg = "Input x must be a one-dimensional 2 element array"
                    raise ValueError(msg)

                cs, sn, r = dlartg(x[0], x[1])

                G = np.array([[cs, sn], [-sn, cs]])
                r = np.array([r, 0])

                R[p, k] = r

                if k < n - 1:  # adjust rest of row
                    R[p, k + 1 :] = G @ R[p, k + 1 :]

            R = R[:-1]  # np.delete(R, -1)

    return R


@njit
def lsq_solve(
    A: np.ndarray,
    b: np.ndarray,
    AA: np.ndarray,
    epsilon: float,
    free_set: list[int],
    deletions_per_loop: int,
    in_binding_set: list[int] | None = None,
    min_val=0,
):
    """Computes a feasible solution non-negative least-squares problem.

    Uses an active-set strategy to handle the non-negativity constraints,
    combined with a preconditioned conjugate gradient solver for the
    unconstrained least-squares problem.
    """
    free_set = sorted(free_set, reverse=True)
    if in_binding_set is not None:
        binding_set = sorted(in_binding_set, reverse=True)
    else:
        binding_set = [0]
        binding_set.remove(0)

    # Reduce A to B.
    # B is a matrix that has all of the rows of A, but its
    # columns are a subset of the columns of A. The free_set
    # provides a map from the columns of B to the columns of A.
    B = A[:, np.array(free_set)].copy()

    # Reduce AA to BB.
    # BB is a symmetric matrix that has a subset of rows and
    # columns of AA. The free_set provides a map from the rows
    # and columns of BB to rows and columns of AA.
    free_set_arr = np.array(free_set)
    num_indices = free_set_arr.shape[0]

    BB = np.empty((num_indices, num_indices), dtype=AA.dtype)

    # Manually fill the BB array using nested loops
    # This directly implements the "cross-product" indexing logic
    for i in range(num_indices):
        row_idx = free_set_arr[i]
        for j in range(num_indices):
            col_idx = free_set_arr[j]
            BB[i, j] = AA[row_idx, col_idx]

    # Cholesky decomposition
    n = AA.shape[0]
    R, p = chol(BB)
    while p > 0:
        # It may be necessary to add to the diagonal of B'B to avoid
        # taking the square root of a negative number when there are
        # rounding errors on a nearly singular matrix. That's still OK
        # because we just use the Cholesky factor as a preconditioner.
        epsilon = epsilon * 10

        AA = AA + (epsilon * np.eye(n))

        free_set_arr = np.array(free_set)
        num_indices = free_set_arr.shape[0]

        BB = np.empty((num_indices, num_indices), dtype=AA.dtype)

        # Manually fill the BB array using nested loops
        # This directly implements the "cross-product" indexing logic
        for i in range(num_indices):
            row_idx = free_set_arr[i]
            for j in range(num_indices):
                col_idx = free_set_arr[j]
                BB[i, j] = AA[row_idx, col_idx]

        R, p = chol(BB)

    # Loops until solution is feasible
    dels = 0
    loops = 0
    lsq_loops = 0
    del_hist = []

    while True:
        loops += 1

        # Use PCGNR to find the unconstrained optimum in the "free" variables.
        reduced_x, k = pcgnr(B, b, R.astype(b.dtype))

        lsq_loops = max(k, lsq_loops)

        # Get a list of variables that must be deleted
        deletion_set = [i for i, _ in enumerate(free_set) if reduced_x[i] <= min_val]

        # If the current solution is feasible then quit.
        if not deletion_set:
            break

        # Sort the possible deletions by their reduced_x values to
        # find the worst violators.
        x_score = reduced_x[np.array(deletion_set)]
        set_index = np.argsort(x_score)
        deletion_set = [deletion_set[i] for i in set_index]

        # Limit the number of deletions per loop
        if len(deletion_set) > deletions_per_loop:
            deletion_set = deletion_set[:deletions_per_loop]

        deletion_set = sorted(deletion_set, reverse=True)
        del_hist.extend(deletion_set)
        dels += len(deletion_set)

        # Move the variables from "free" to "binding".
        # bound_variables = free_set[deletion_set]
        bound_variables = [free_set[i] for i in deletion_set]
        if binding_set is not None:
            binding_set.extend(bound_variables)
        else:
            binding_set = bound_variables
        free_set = [j for j in free_set if j not in bound_variables]
        if len(free_set) == 0:
            break
        # Reduce A to B
        # B is a matrix that has all of the rows of A, but its
        # columns are a subset of the columns of A. The free_set
        # provides a map from the columns of B to the columns of A.
        B = A[:, np.array(free_set)].copy()

        # Reduce AA to BB.
        # BB is a symmetric matrix that has a subset of rows and
        # columns of AA. The free_set provides a map from the rows
        # and columns of BB to rows and columns of AA.
        free_set_arr = np.array(free_set)
        num_indices = free_set_arr.shape[0]

        BB = np.empty((num_indices, num_indices), dtype=AA.dtype)

        # Manually fill the BB array using nested loops
        # This directly implements the "cross-product" indexing logic
        for i in range(num_indices):
            row_idx = free_set_arr[i]
            for j in range(num_indices):
                col_idx = free_set_arr[j]
                BB[i, j] = AA[row_idx, col_idx]
        # Compute R, the Cholesky factor.
        R = _cholesky_delete(R, BB, np.array(deletion_set))

    # Unscramble the column indices to get the full (unreduced) x.
    n = A.shape[1]
    x = np.zeros(n, dtype=A.dtype)
    if len(free_set) != 0:
        x[np.array(free_set)] = reduced_x

    # Compute the full (unreduced) residual.
    residual = b - (A @ x)

    # Compute the norm of the residual.
    score = np.sqrt(np.dot(residual, residual))

    return (
        score,
        x,
        residual,
        free_set,
        binding_set,
        AA,
        epsilon,
        del_hist,
        dels,
        loops,
        lsq_loops,
    )


@guvectorize(
    [
        "void(float32[:,::1], float32[::1], float32[:,::1], float32, float32, float32, float32, float32[::1])",
        "void(float64[:,::1], float64[::1], float64[:,::1], float32, float32, float32,float32, float64[::1])",
    ],
    "(m,n),(m),(n,n),(),(),(),(),(n)",
    forceobj=True,
    **nb_kwargs,
)
def tntnn(
    A: np.ndarray,
    b: np.ndarray,
    AA: np.ndarray | None,
    rel_tol: float,
    red_c: float,
    exp_c: float,
    min_val: float,
    x: np.ndarray,
):
    """
    Example values: rel_tol=0, red_c=0.2, exp_c = 1.2
    """

    hist = []

    m, n = A.shape

    if A.dtype != AA.dtype:
        msg = "A and AA must have same dtype."
        raise ValueError(msg)

    if b.shape != (m,):
        msg = "A must have the same number of rows as the dimension of b."
        raise ValueError(msg)

    # AA is a symmetric and positive definite (probably) n x n matrix.
    # If A did not have full rank, then AA is positive semi-definite.
    # Also, if A is very ill-conditioned, then rounding errors can make
    # AA appear to be indefinite. Modify AA a little to make it more
    # positive definite.
    epsilon = 10 * np.spacing(1) * la.norm(AA, 1)
    AA = AA + (epsilon * np.eye(n))

    # In this routine A will never be changed, but AA might be adjusted
    # with a larger "epsilon" if needed. Working copies called B and BB
    # will be used to perform the computations using the "free" set
    # of variables.

    # Initialize sets.
    free_set = list(range(n))
    binding_set = None

    # This sets up the unconstrained, core LS solver
    score, x[:], residual, free_set, binding_set, AA, epsilon, _, dels, lps, _ = lsq_solve(
        A, b, AA, epsilon, free_set, n, in_binding_set=binding_set, min_val=min_val
    )

    # Outer loop
    OuterLoop = 0
    TotalInnerLoops = 0
    insertions = n
    continue_outer_loop = True  # used to return from inner loop

    while continue_outer_loop:

        OuterLoop += 1

        # Save this solution
        best_score = score
        best_x = x.copy()
        best_free_set = free_set[:]
        best_binding_set = binding_set[:]
        best_insertions = insertions
        max_insertions = math.floor(exp_c * best_insertions)

        # Compute the gradient of the "Normal Equations"
        gradient = A.T @ residual

        # Check the gradient components
        insertion_set = [i for i, bi in enumerate(binding_set) if gradient[bi] > 0]
        insertions = len(insertion_set)

        # Are we done ?
        if insertions == 0:
            # There were no changes that were feasible.
            # We are done.
            hist.append([0 for i in range(6)])
            break

        # Sort the possible insertions by their gradients to find the
        # most attractive variables to insert.
        grad_score_coordinates = [binding_set[i] for i in insertion_set]
        grad_score = gradient[grad_score_coordinates]
        set_index = np.argsort(grad_score)[::-1]  # use descending order
        insertion_set = [insertion_set[i] for i in set_index]

        # Inner loop
        InnerLoop = 0
        while True:

            InnerLoop += 1
            TotalInnerLoops += 1

            # Adjust the number of insertions.
            insertions = math.floor(red_c * insertions)
            if insertions == 0:
                insertions = 1
            insertions = min(insertions, max_insertions)
            insertion_set = insertion_set[:insertions]

            # Move variables from "binding" to "free"
            free_variables = [binding_set[i] for i in insertion_set]
            free_set.extend(free_variables)
            binding_set = [j for j in binding_set if j not in free_variables]

            # Compute a feasible solution using the unconstrained
            # least-squares solver of your choice.
            (
                score,
                x[:],
                residual,
                free_set,
                binding_set,
                AA,
                epsilon,
                _,
                dels,
                lps,
                _,
            ) = lsq_solve(
                A,
                b,
                AA,
                epsilon,
                free_set,
                insertions,
                in_binding_set=binding_set,
                min_val=min_val,
            )

            # Check for new best solution
            if score < best_score * (1 - rel_tol):
                break

            # Restore the best solution
            score = best_score
            x[:] = best_x.copy()
            free_set = best_free_set[:]
            binding_set = best_binding_set[:]
            max_insertions = math.floor(exp_c * best_insertions)

            # Are we done ?
            if insertions == 1:
                # The best feasible change did not improve the score.
                # We are done.
                continue_outer_loop = False
                break
