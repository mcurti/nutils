# Plane strain plate under gravitational pull
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the top boundary, and stretched under the influence of a
# vertical distributed load.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy

def main(nelems: int, etype: str, btype: str, degree: int, poisson: float, direct: bool):
    '''
    Horizontally loaded linear elastic plate.

    .. arguments::

       nelems [24]
         Number of elements along edge.
       etype [triangle]
         Type of elements (square/triangle/mixed).
       btype [std]
         Type of basis function (std/spline), with availability depending on the
         configured element type.
       degree [2]
         Polynomial degree.
       poisson [.3]
         Poisson's ratio, nonnegative and strictly smaller than 1/2.
       direct [no]
         Use direct traction evaluation.
    '''

    domain, geom = mesh.unitsquare(nelems, etype)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

    basis = domain.basis(btype, degree=degree)
    ns.u = function.dotarg('u', basis, shape=(2,))
    ns.X_i = 'x_i + u_i'
    ns.λ = 1
    ns.μ = .5/poisson - 1
    ns.ε_ij = '.5 (∇_i(u_j) + ∇_j(u_i))'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.E = 'ε_ij σ_ij'
    ns.q_i = '-δ_i1'

    sqr = domain.boundary['top'].integral('u_k u_k dS' @ ns, degree=degree*2)
    cons = solver.optimize(('u',), sqr, droptol=1e-15)

    # solve for equilibrium configuration
    internal = domain.integral('E dV' @ ns, degree=degree*2)
    external = domain.integral('u_i q_i dV' @ ns, degree=degree*2)
    args = solver.optimize(('u',), internal - external, constrain=cons)

    # evaluate tractions and net force
    if direct:
        ns.t_i = 'σ_ij n_j' # <-- this is an inadmissible boundary term
    else:
        ns.t = function.dotarg('t', basis, shape=(2,))
        external += domain.boundary['top'].integral('u_i t_i dS' @ ns, degree=degree*2)
        invcons = dict(t=numpy.choose(numpy.isnan(cons['u']), [numpy.nan, 0.]))
        args = solver.solve_linear(('t',), [(internal - external).derivative('u')], constrain=invcons, arguments=args)
    F = domain.boundary['top'].integrate('t_i dS' @ ns, degree=degree*2, arguments=args)
    log.user('total clamping force:', F)

    # visualize solution
    bezier = domain.sample('bezier', 3)
    X, E = bezier.eval(['X_i', 'E'] @ ns, **args)
    Xt, t = domain.boundary['top'].sample('bezier', 2).eval(['X_i', 't_i'] @ ns, **args)
    with export.mplfigure('energy.png') as fig:
        ax = fig.add_subplot(111, ylim=(-.2,1), aspect='equal')
        im = ax.tripcolor(*X.T, bezier.tri, E, shading='gouraud', rasterized=True, cmap='turbo')
        export.plotlines_(ax, X.T, bezier.hull, colors='k', linewidths=.1, alpha=.5)
        ax.quiver(*Xt.T, *t.T, clip_on=False)
        fig.colorbar(im)

    return cons, args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

    @testing.requires('matplotlib')
    def test_default(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYMACGqgLASCRFAE=''')
        with self.subTest('displacement'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYMAEBYYKBkqGMXqyhgwMSoZLLhYYPji/wajBYI6Rjv4kIwaGOUZXLgD550uNpxvkG7voZxszMOQb
                77lQapx5ns1kkQGzSZA+owkDA7PJugtsJnHnATXSGpw=''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNoTObHnJPvJeoP9JxgY2E82nhc54WLGQGUAACftCN4=''')

    @testing.requires('matplotlib')
    def test_mixed(self):
        cons, args = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYICCBiiEsdFpCiAARJEUAQ==''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYICAPEMhAy1DBT0Qm9vwnDqI1jW8dBFE5xi+Oz/LSEt/s1G5wUSjyTdmGD25sMmo/3yZ8UyDfGMn
                /UzjJ6p5xsculBrnnGc2idNnN1lmwGDCpcZksuECm0nCeQD9cB5S''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNrjPXH7pMbJJ+cZoGDyCYvLIFr7pJEBiOY+oW3GQCEAAGUgCg4=''')

    @testing.requires('matplotlib')
    def test_quadratic(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=2, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYCACNIxCfBAAg5xIAQ==''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNqFzT9LwlEUxvFLIj/FQQjLftGmtNj1nHvOHZqK8A0EvoKWwCEFQRRaWsJaazBbkja3AufEbGi711tT
                1Np/2oSUiO476Fk/X3iE+H9N/IQihPketGQbHnPnIEQbsvc9KLkivIyamLINlcaCagCo3XxWTVYySois
                Cu5A7Y8K6sD7m8lRHdP0BHGahak6lT++maptF6cvm6aMzdGhuaQ97NIYOrQMbbqVJ+S/aNV16MF2KWG9
                myU+wpADTPEaJPlZJlmIJC+6FF/bkCfey6bOx1jjGFZ5HSr8Ksu+qfCCq/LA1vjb+4654RYOOYED3oA+
                v8sr3/R53g24b4c89l4yMX2GgZ7DqN6EiP6VES1ERM+4qL6wgf7wvmX+AN5xajA=''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpbejz8pNcpFdO5J26d5jy5y+DwCQYGzpNu5+eeUDPxOnXn1NLjK80YRgFeAAC0chL2''')

    @testing.requires('matplotlib')
    def test_poisson(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=1, poisson=.4)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYMACGqgLASCRFAE=''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYMAEOcZHje8byRhdN2JguG/05GyOsfWZkyYyJnNMzhl1mDAwzDExOnvS5MnpmaYmJp2mj4waTBkY
                Ok3lzs40PXPayMzRRMvsg5GaGQODlpnAWSOz/acBAbAecQ==''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNrbdFz7ROrJs8aHTjAwpJ40PrPp+FVzBioDANbTCtc=''')

# example:tags=elasticity
