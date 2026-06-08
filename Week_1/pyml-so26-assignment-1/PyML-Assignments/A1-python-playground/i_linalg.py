"""
Fill out the code below so that they perform the mathematical operations
that are described in the docstring.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""
import math

def gram_schmidt(vectors: list[list[float]]) -> list[list[float]]:
    """
    Orthonormalizes a set of linearly independent vectors using the
    Gram-Schmidt process.

    More about this method can be found here:

        https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    TIP: Write several helper functions within the body of this function.

    Arguments:
        vectors -- a list of linearly independent vectors, each represented
            as a list of floats of equal length

    Returns:
        list[list[float]] -- a list of orthonormal vectors spanning the same
            subspace as the input vectors

    Raises:
        ValueError -- if the vectors are not all the same length
        ValueError -- if the vectors are linearly dependent
    """

    # ------ SOLUTION GOES HERE!  ------
    def dot(u, v):
        return sum(ui * vi for ui, vi in zip(u, v))

    def norm(v):
        return math.sqrt(dot(v, v))

    def proj(u, v):
        """Projection of u onto v."""
        denom = dot(v, v)
        if denom == 0:
            raise ValueError("Vectors are linearly dependent")
        scale = dot(u, v) / denom
        return [scale * vi for vi in v]

    # --- dimension check ---
    if len(vectors) == 0:
        return []

    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("All vectors must have the same dimension")

    # --- Gram-Schmidt ---
    orthonormal = []
    for v in vectors:
        w = v[:]
        for u in orthonormal:
            p = proj(w, u)
            w = [wi - pi for wi, pi in zip(w, p)]

        n = norm(w)
        if n == 0:
            raise ValueError("Vectors are linearly dependent")

        orthonormal.append([wi / n for wi in w])

    return orthonormal

def gaussian_elimination(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Solves the linear system Ax = b using Gaussian elimination
    with partial pivoting.

    More about this method can be found here:

        https://en.wikipedia.org/wiki/Gaussian_elimination

    Arguments:
        A -- an n x n matrix represented as a list of lists of floats
        b -- a list of n floats representing the right-hand side vector

    Returns:
        list[float] -- the solution vector x such that Ax = b

    Raises:
        ValueError -- if A is not square or dimensions are incompatible
        ValueError -- if the system is singular (no unique solution exists)

    (_Hint: Partial pivoting — before eliminating column `col`, find the row
    in `range(col, n)` with the largest absolute value in that column and swap
    it into the pivot position. This keeps the algorithm numerically stable._)
    """

    # ------ SOLUTION GOES HERE!  ------
    n = len(A)

    # --- dimension checks ---
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a non-empty square matrix")
    if len(b) != n:
        raise ValueError("Dimensions of A and b do not match")

    # --- build augmented matrix ---
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # --- forward elimination with partial pivoting ---
    for col in range(n):
        # pivot: row with largest |value|
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular")

        # swap
        M[col], M[pivot_row] = M[pivot_row], M[col]

        # eliminate rows below
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # --- back substitution ---
    x = [0.0] * n
    for i in reversed(range(n)):
        if abs(M[i][i]) < 1e-12:
            raise ValueError("Matrix is singular")

        rhs = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = rhs / M[i][i]

    return x

if __name__ == "__main__":
    # Orthonormalize two non-orthogonal vectors in R^2
    basis = gram_schmidt([[3, 0], [1, 1]])
    print("gram_schmidt:", [f"[{', '.join(f'{x:.4f}' for x in v)}]" for v in basis])

    # Solve 2x + y = 5, x + 3y = 10  →  x=1.0, y=3.0
    x = gaussian_elimination([[2, 1], [1, 3]], [5, 10])
    print("gaussian_elimination:", [round(v, 4) for v in x])
