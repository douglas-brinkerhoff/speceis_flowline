Width-parameterized version of the Spectral Ice Sheet Model (SpecEIS).  This model solves the first order equations of ice sheet flow, using 4th order polynomials (plus a constant) as test and trial functions for the finite element method.  This yields three velocity coefficients: the depth and width averaged velocity, and the deformational velocity in the vertical and horizontal dimensions.  These equations are coupled to a 0-order Discontinuous Galerkin method for solving the flux equations, and rounded off with a global mass conservation equation for glacier length.  The equations are solved in a coupled and fully implicit fashion, with positivity constraints enforced through the PETSc VI SNES solver.  