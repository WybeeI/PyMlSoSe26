"""
Fill out the code below so that they perform the mathematical operations
that are described in the docstring.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""


def quadratic_formula(a: float, b: float, c: float) -> tuple:
    """
    Calculates the roots of a quadratic equation given the
    coefficients a, b, and c.

    Arguments:
        a -- the coefficient of x^2
        b -- the coefficient of x
        c -- the constant term

    Returns:
        tuple(root1, root2) -- the real roots (no complex roots) of
            the quadratic equation in numerical order (i.e. root1 <= root2).

        tuple(root1,) -- if there is only one real root.

        tuple() -- an empty tuple, if there are no real roots.
    """

    # ------ SOLUTION GOES HERE!  ------
    # Discriminante
    disc = b**2 - 4 * a * c

    if disc < 0:
        return ()  # No hay raíces reales

    if disc == 0:
        root = -b / (2 * a)
        return (root,)

    # Dos raíces reales
    sqrt_disc = disc**0.5
    r1 = (-b - sqrt_disc) / (2 * a)
    r2 = (-b + sqrt_disc) / (2 * a)

    return (min(r1, r2), max(r1, r2))


def simpsons_rule(f, a: float, b: float, n: int) -> float:
    """
    Approximates the definite integral of f from a to b using
    composite Simpson's 1/3 rule.

    https://en.wikipedia.org/wiki/Simpson%27s_rule

    Arguments:
        f -- a callable f(x) representing the integrand
        a -- the lower bound of integration
        b -- the upper bound of integration
        n -- the number of subintervals (must be a positive even integer)

    Returns:
        float -- the approximate value of the integral

    Raises:
        ValueError -- if n is not a positive even integer
    """

    # ------ SOLUTION GOES HERE!  ------
    # n debe ser par y positivo
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer")

    h = (b - a) / n
    total = f(a) + f(b)

    # Suma de términos impares (coef 4)
    odd_sum = 0
    for i in range(1, n, 2):
        odd_sum += f(a + i * h)

    # Suma de términos pares (coef 2)
    even_sum = 0
    for i in range(2, n, 2):
        even_sum += f(a + i * h)

    return (h / 3) * (total + 4 * odd_sum + 2 * even_sum)


def matrix_mult(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Calculates the matrix product AB.

    Arguments:
        A -- A list of lists of floats, representing a matrix, with each
            element of the outer list being a row.
        B -- Another matrix of the same form.

    Returns:
        The matrix product AB.

    Raises:
        ValueError -- If the number of columns of A does not match the number
            of rows in B.
    """

    # ------ SOLUTION GOES HERE!  ------
    # Dimensiones
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Number of columns of A must match number of rows of B")

    # Crear matriz resultado
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Multiplicación estándar
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


if __name__ == "__main__":
    # x^2 - 5x + 6 = 0  →  roots (2.0, 3.0)
    print("quadratic_formula:", quadratic_formula(1, -5, 6))

    # integral of x^2 from 0 to 1  →  ≈ 0.3333
    print("simpsons_rule:    ", simpsons_rule(lambda x: x**2, 0, 1, 4))

    # [[1,2],[3,4]] @ [[5,6],[7,8]]  →  [[19,22],[43,50]]
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print("matrix_mult:      ", matrix_mult(A, B))
