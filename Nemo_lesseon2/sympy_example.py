from sympy import symbols, Integral, sin, cos, exp, oo, pi, init_printing

# initializa LaTeX representation
init_printing()

# define symbolic variable
x, y, a = symbols('x y a')

# define equations
expr1 = Integral(sin(x)**2, (x, 0, pi))
expr2 = Integral(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo))
expr3 = Integral(exp(-x**2), (x, -oo, oo))
expr4 = Integral(cos(a * x), x)
expr5 = Integral(x * exp(x), x)

# print
print(f"∫ sin²(x) dx from 0 to π = {expr1.doit()}")
print(f"∬ e^-(x² + y²) dx dy from -∞ to ∞ = {expr2.doit()}")
print(f"∫ e^-x² dx from -∞ to ∞ = {expr3.doit()}")
print(f"∫ cos(a·x) dx = {expr4.doit()}")
print(f"∫ x·e^x dx = {expr5.doit()}")