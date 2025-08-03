from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec

ns = NormalDotVec(["u", "v", "w"])
ns.pprint()
ns2 = NavierStokes(nu=0.025 * 0.4, rho=1.0, dim=3, time=False)
ns2.pprint()