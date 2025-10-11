import numpy as np
import sympy as sp

def make_symbolic_2link():
    q1, q2 = sp.symbols('q1 q2', real=True)
    dq1, dq2 = sp.symbols('dq1 dq2', real=True)
    l1, l2, m1, m2, I1, I2, g = sp.symbols('l1 l2 m1 m2 I1 I2 g', positive=True, real=True)
    q = sp.Matrix([q1,q2]); dq = sp.Matrix([dq1,dq2])
    # COM positions
    x1 = (l1/2)*sp.cos(q1); y1 = (l1/2)*sp.sin(q1)
    x2 = l1*sp.cos(q1) + (l2/2)*sp.cos(q1+q2); y2 = l1*sp.sin(q1) + (l2/2)*sp.sin(q1+q2)
    J1 = sp.Matrix([x1,y1]).jacobian(q); J2 = sp.Matrix([x2,y2]).jacobian(q)
    v1 = J1 * dq; v2 = J2 * dq
    omega1 = dq1; omega2 = dq1 + dq2
    T1 = sp.Rational(1,2)*m1*(v1.dot(v1)) + sp.Rational(1,2)*I1*omega1**2
    T2 = sp.Rational(1,2)*m2*(v2.dot(v2)) + sp.Rational(1,2)*I2*omega2**2
    T = sp.simplify(T1 + T2)
    V = sp.simplify(m1*g*y1 + m2*g*y2)
    return dict(q1=q1,q2=q2,dq1=dq1,dq2=dq2,l1=l1,l2=l2,m1=m1,m2=m2,I1=I1,I2=I2,g=g,
                q=q,dq=dq,T=T,V=V,J1=J1,J2=J2)

def lambdify_dynamics(M_s, C_s, G_s):
    q1, q2, dq1, dq2, l1, l2, m1, m2, I1, I2, g = sp.symbols('q1 q2 dq1 dq2 l1 l2 m1 m2 I1 I2 g')
    Mf = sp.lambdify((q1,q2,l1,l2,m1,m2,I1,I2), M_s, 'numpy')
    Cf = sp.lambdify((q1,q2,dq1,dq2,l1,l2,m1,m2,I1,I2), C_s, 'numpy')
    Gf = sp.lambdify((q1,q2,l1,l2,m1,m2,g), G_s, 'numpy')
    return Mf, Cf, Gf
