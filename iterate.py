import numpy as np

# Find polynomial for arctan


def error(x, vec):
    a, b, c, d = vec
    # return (c * x + (b * x ** 2) + (a * x ** 4)) - (1.0 - np.sqrt(1.0 - x ** 2))
    return ((c * x + (a * x ** 2) + (b * x ** 4)) - np.arctan(x) * 1.273239) ** 2.0


def error_gradient(x, vec):
    a, b, c, d = vec
    if x > 0.0:
        t = 1.273239 / (x ** 2 + 1.0)
    else:
        t = 0.0
    return (c + 2 * a * x + 4 * b * x ** 3) - t


def sum_sqr(term):
    def _foo(vec):
        return sum(term(x / 100, vec) for x in range(100))

    return _foo


def descent(vec, grad, itn=10000):
    max_speed = 0.1
    pstep = np.zeros_like(vec)
    for i in range(itn):
        gvec = grad(vec)
        if gvec < 0.0002:
            print("Broke at:", i)
            break

        nice = (gvec * 200.0) ** 2.0
        if nice > max_speed:
            nice = max_speed

        step = (np.random.rand(vec.shape[0]) - 0.5 + pstep) * nice
        pstep *= 0.95
        if gvec - grad(vec + step) > 0.0:
            vec += step
            pstep = step

        if i % 1000 == 0:
            print(i, vec, gvec, nice)
    return vec


# np.array([0.0, 0.0, 0.0, 0.0]
# np.array([-0.272, -0.052, 1.32])
vec = descent(np.array([0.0, 0.0, 1.0, 0.0]), sum_sqr(error))
print("Verify.")
print(sum_sqr(error)(vec))
print(vec)

# Do it with numpy
print("---")
x = np.arange(100) / 100.0
y = np.arctan(x) * 1.273239
coeffs = np.polynomial.polynomial.polyfit(x, y, deg=(1, 2, 4))
print(coeffs)

# assert abs(squared_error(1.0, 0.0, 0.0, 1.0)) < 0.00001

# for x in range(100):
#     e = abs(error(x / 100, 0.0, 0.0, 1.1))
#     # print(x / 100, e)
#     assert e < 0.1

# assert sum(squared_error(x / 100, -0.35, 0.0, 1.35) for x in range(100)) < sum(
#     squared_error(x / 100, 0.0, 0.0, 1.09) for x in range(100)
# )
