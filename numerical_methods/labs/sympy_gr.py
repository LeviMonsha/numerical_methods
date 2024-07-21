# import SymPy
import sympy
from sympy import *
from sympy.abc import *
from sympy.plotting import *

gr = plot(5*sin(x), (x,-5,5), line_color="r",title="Graph",xlabel="Axis X",ylabel="Axis Y")

q = plot(10*sin(x), exp(x), 25*log(x), (x,-5,5), show=False)
q[1].line_color="r"
q[2].line_color="g"
gr.show()

# x = Symbol('x')
# print(sympy.integrate(16*x, x))
# print(sympy.Rational(3.54).limit_denominator(100))
# t = sympy.Rational(3,19)
# print("числитель:{}, знаменатель:{}".format(t.p, t.q))
# from sympy.abc import x
# from sympy import sin, pi
# expr = sin(x)
# ts = expr.subs(x,pi/6)
# print(ts)
# print(sympify("1/2 + 3/10", evaluate=true))