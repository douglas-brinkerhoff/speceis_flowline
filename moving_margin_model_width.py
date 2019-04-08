import dolfin as df
import matplotlib 
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['quadrature_degree'] = 4
df.parameters['allow_extrapolation'] = True

ffc_options = {"optimize": True}

# HELPER FUNCTIONS
def softplus(y1,y2,alpha=1):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return y1 + (1./alpha)*df.ln(1+df.exp(alpha*(y2-y1)))

##########################################################
###############        CONSTANTS       ###################
##########################################################

rho = 917.                  # Ice density
rho_w = 1029.0              # Seawater density

g = 9.81                    # Gravitational acceleration
n = 3.0                     # Glen's exponent
A = 1e-17
b = A**(-1./n)              # Ice hardness
eps_reg = 1e-5              # Regularization parameter

spy = 60**2*24*365          # Seconds per year

#########################################################
################    PARAMETERS    #######################
#########################################################

# Time interval
t_start = 0.0
t_end = 12000.
dt_max = 10.0          # Maximum time step
theta_dt = 0.01       # After time step is adaptively reduced, it
                      # recovers to the maximum value according to 
                      # dt_float = theta_dt*dt_max + (1-theta_dt)*dt_float
dt_float = 0.01*dt_max     # initial time step guess 


theta_H = 0.5         # implicitness for thickness solve
theta_L = 0.5         # implicitness for length solve 
# Note that the above values get overruled in favor of backward Euler
# for part of the length equation to avoid small oscillations

L_0 = 200000.           # Characteristic domain length
zmax = 500.0            # Maximum elevation
slope = zmax/(0.33*L_0) # Average Bed slope

corr_len = 5000.0       # Correlation length of random topography
rand_amp = 35.0         # Amplitude of random topography

wmax = 5.0              # Maximum width
wslope = 1.0            # Width exponent

amax = 1.0              # Surface mass balance at ice divide

thklim = 1.0            # Minimum ice thickness (req.
                        # for numerical stability - ostensibly, 
                        # the thickness never gets below 0 in this model

L_initial = 6.5e4       # Initial guess for length.  Matters alot for the 
                        # local length change model, little for the 
                        # integral form.

q = 0.0                 # Fraction above flotation where calving begins
                        # q = 0 -> calving at flotation,
                        # q = 0.15 -> obs. at Columbia Gl.            

k_calving = 10.0        # Strength of the calving rate for integral length
                        # model.  As k -> infty, boundary condition becomes
                        # Dirichlet, H=H_c 

backstress = 0          # Additional backstress imposed at the terminus

alpha2 = 1e3

PLOT = True             # Plot or not
STOREALL = False        # Save full thickness and velocity solutions
PLOT_INTERVAL = 50      # n years between plot updates

#########################################################
#################      GEOMETRY     #####################
#########################################################

# Generate random topography modelled as a gaussian random process with 
# a squared exponential spatial correlation function, correlation length
# corr_len, and amplitude rand_amp.  This is then superimposed upon 
# a rightward sloping bed that has max elevation zmax.  The width is 
# determined as exponentially decreasing from wmax according to an exponent
# wslope, but this is specified as a form to make computing dW/dt possible
# implicitly.

# Random topography
np.random.seed(1234)
x_random = np.linspace(0,2*L_0,801)
N_random = len(x_random)
X,_ = np.meshgrid(x_random,x_random)
cov = rand_amp**2*np.exp(-((X-X.T)**2/corr_len**2))
z_noise = np.random.multivariate_normal(np.zeros(N_random),cov)
iii = spi.interp1d(x_random,z_noise)

# Bed elevation
class Bed(df.UserExpression):
  def __init__(self,L_initial,**kwargs):
    self.L = L_initial
    super().__init__(**kwargs)
  def eval(self,values,x):
    values[0] = zmax - slope*x[0]*self.L + iii(x[0]*self.L)

# Flowline width
class Width(df.UserExpression):
  def __init__(self,L_initial,**kwargs):
    self.L = L_initial
    super().__init__(**kwargs)
  def eval(self,values,x):
    values[0] =  wmax*np.exp(-wslope*x[0]*self.L/L_0)

#########################################################
####################  MASS BALANCE  #####################
#########################################################

# Mass balance linearly decreases from amax according to the 
# slope that will yield a terminus position at L

class Adot(df.UserExpression):
  def __init__(self,L_initial,**kwargs):
    self.L = L_initial
    super().__init__(**kwargs)
  def eval(self,values,x):
    c = amax*wslope*(np.exp(wslope)-1)/((np.exp(wslope)-wslope-1)*L_0)
    values[0] = amax - c*x[0]*self.L

##########################################################
###################  BASAL TRACTION  #####################
##########################################################

# Constant basal traction

class Beta2(df.UserExpression):
  def eval(self,values,x):
    values[0] = 1e-3

##########################################################
########################  MESH  ##########################
##########################################################

# Define a unit interval mesh with equal cell size
N_cells = 300
mesh = df.IntervalMesh(N_cells,0,1)

# Shift vertices such that they are concentrated towards the 
# terminus following a polynomial curve.
mesh_exp = 1.5   
mesh.coordinates()[:] = (1-mesh.coordinates()**mesh_exp)[::-1]

# Mesh Functions
h = df.CellDiameter(mesh)                   # Cell size
x_spatial = df.SpatialCoordinate(mesh)  # spatial coordinate
nhat = df.FacetNormal(mesh)             # facet normal vector
ocean = df.MeshFunction('size_t',mesh,0,0)   # boundary subdomain function
                                        # ocean=1 -> terminus
                                        # ocean=2 -> ice divide
df.ds = df.ds(subdomain_data=ocean)

for f in df.facets(mesh):
    if df.near(f.midpoint().x(),1):
       ocean[f] = 1
    if df.near(f.midpoint().x(),0):
       ocean[f] = 2

#########################################################
#################  FUNCTION SPACES  #####################
#########################################################

# Define finite element function spaces.  Here we use CG1 for 
# velocity computations, DG0 (aka finite volume) for mass cons,
# and "Real" (aka constant) elements for the length ODE 

E_Q = df.FiniteElement("CG",mesh.ufl_cell(),1)
E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
E_R = df.FiniteElement("R",mesh.ufl_cell(),0)
E_V = df.MixedElement(E_Q,E_Q,E_Q,E_Q,E_dg,E_R)

Q = df.FunctionSpace(mesh,E_Q)
Q_dg = df.FunctionSpace(mesh,E_dg)
Q_R = df.FunctionSpace(mesh,E_R)
V = df.FunctionSpace(mesh,E_V)

# For moving data between vector functions and scalar functions 
assigner_inv = df.FunctionAssigner([Q,Q,Q,Q,Q_dg,Q_R],V)
assigner     = df.FunctionAssigner(V,[Q,Q,Q,Q,Q_dg,Q_R])

#########################################################
#################  FUNCTIONS  ###########################
#########################################################

# The zero function
ze = df.Function(Q)           

# U contains both velocity components, the DG thickness, 
# the CG-projected thickness, and the length
U = df.Function(V)
dU = df.TrialFunction(V)        # Trial Function
Phi = df.TestFunction(V)        # Test Function

# Split vector functions into scalar components
ubar,udef,uwid,H_c,H,L = df.split(U)
phibar,phidef,phiwid,xsi_c,xsi,chi = df.split(Phi)

# Values of model variables at previous time step
un = df.Function(Q)
u2n = df.Function(Q)
u3n = df.Function(Q)
H0_c = df.Function(Q)
H0 = df.Function(Q_dg)
L0 = df.Function(Q_R)

Bhat = df.Function(Q)          # The bed topography
beta2 = df.Function(Q)         # The basal traction
adot = df.Function(Q)          # The surface mass balance

l = softplus(df.Constant(0),Bhat)           # Water surface, or the greater of
                                            # bedrock topography or zero

B = softplus(Bhat,-rho/rho_w*H_c,alpha=0.2) # Ice base is the greater of the 
                                            # bedrock topography or the base of 
                                            # the shelf

D = softplus(-Bhat,df.Constant(0))          # Water depth
S = B + H_c                                 # Ice surface = Ice base + thickness

####################################################
############  FUNCTION INITIALIZATION  #############
####################################################

# Initialize thickness and length to reasonable values.  It is particularly
# important that dHdx != 0 for the local length change method to work.
# Doesn't matter much for the integrated form.
 
#H0.vector()[:] = thklim
H0.interpolate(df.project(55 - 55*x_spatial[0],Q_dg))
H0.vector()[:] += thklim
H0_c.interpolate(H0)
L0.vector()[:] = L_initial

assigner.assign(U,[ze,ze,ze,H0_c,H0,L0])  # Update U with these values

# Interpolate expressions for bedrock, traction, mass balance
Bhat.interpolate(Bed(L_initial,degree=1))
beta2.interpolate(Beta2(degree=1))
adot.interpolate(Adot(L_initial,degree=1))

# Bounds for snes_vi_rsls.  Only thickness bound is ever used.
l_v_bound = df.interpolate(df.Constant(-1e10),Q)
u_v_bound = df.interpolate(df.Constant(1e10),Q)

l_thickc_bound = df.interpolate(df.Constant(thklim),Q)
u_thickc_bound = df.interpolate(df.Constant(1e10),Q)

l_thick_bound = df.interpolate(df.Constant(thklim),Q_dg)
u_thick_bound = df.interpolate(df.Constant(1e10),Q_dg)

l_r_bound = df.interpolate(df.Constant(-1e10),Q_R)
u_r_bound = df.interpolate(df.Constant(1e10),Q_R)

l_bound = df.Function(V)
u_bound = df.Function(V)

assigner.assign(l_bound,[l_v_bound]*3+[l_thickc_bound]+[l_thick_bound]+[l_r_bound])
assigner.assign(u_bound,[u_v_bound]*3+[u_thickc_bound]+[u_thick_bound]+[u_r_bound])

###############################################
################ TIME STEPPING  ###############
###############################################

# This gets changed according to dt_schedule
dt = df.Constant(0.0)

Hmid = df.Constant(theta_H)*H + df.Constant(1-theta_H)*H0
Lmid = df.Constant(theta_L)*L + df.Constant(1-theta_L)*L0

# Width expressions
width_n = wmax*df.exp(-wslope*x_spatial[0]*L/L_0)
width_0 = wmax*df.exp(-wslope*x_spatial[0]*L0/L_0)
width = wmax*df.exp(-wslope*x_spatial[0]*Lmid/L_0)

# Time derivatives
dLdt = (L-L0)/dt
dHdt = (H-H0)/dt
dWdt = (width_n - width_0)/dt

########################################################
#################   MOMENTUM BALANCE   #################
########################################################

# Solves the first-order equations of ice sheet motion using CG1 finite elements
# in the horizontal dimension, and a zero and n+1 order polynomial in the 
# vertical

class FlowlineBasis(object):
    """ 
    Provides dolfin-like access to width and vertical derivatives.  Accepts
    nodal values (u), a list of test functions (coef), and their
    vertical derivatives (dcoef)
    """
    def __init__(self,u,coef,dcoef_s,dcoef_g):
        self.u = u
        self.coef = coef
        self.dcoef_s = dcoef_s
        self.dcoef_g = dcoef_g

    def __call__(self,g,s):
        return sum([u*c(g,s) for u,c in zip(self.u,self.coef)])
    
    def dg(self,g,s):
        return sum([u*c(g,s) for u,c in zip(self.u,self.dcoef_g)])

    def ds(self,g,s):
        return sum([u*c(g,s) for u,c in zip(self.u,self.dcoef_s)])

    def dx(self,g,s,dim):
        return sum([u.dx(dim)*c(g,s) for u,c in zip(self.u,self.coef)])

class FlowlineIntegrator(object):
    """
    Integrates a form in the vertical dimension
    """
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,g,s,w):
        return w*f(g,s)
    def intz(self,f):
        return sum([self.integral_term(f,s[0],s[1],w) for s,w in zip(self.points,self.weights)])

# Sigma-coordinate jacobian terms
def dsdx(g,s):
    return 1./(H_c*Lmid)*(S.dx(0) - s*H_c.dx(0))

def dsdy(g,s):
    return 0

def dsdz(g,s):
    return -1./H_c

def dgdx(g,s):
    return -width.dx(0)*g/(width*Lmid)

def dgdy(g,s):
    return 2./width

def dgdz(g,s):
    return 0.

def dxdx(g,s):
    return 1./Lmid

def dxdy(g,s):
    return 0.

def dxdz(g,s):
    return 0.

# vertical test functions, in this case a constant and a n+1 order polynomial    
coef = [lambda g,s:1.0, lambda g,s:1./4.*(5*s**4-1.), lambda g,s: g**4.]
dcoef_s = [lambda g,s:0.0, lambda g,s:5*s**3, lambda g,s:0]
dcoef_g = [lambda g,s:0.0, lambda g,s:0, lambda g,s:4*g**3]

# Make vertical basis from ubar and udef, the depth-average and 
# deformational velocities
u_ = [ubar,udef,uwid]
phi_ = [phibar,phidef,phiwid]

u = FlowlineBasis(u_,coef,dcoef_s,dcoef_g)
phi = FlowlineBasis(phi_,coef,dcoef_s,dcoef_g)

def dudx(g,s):
    return u.dx(g,s,0)*dxdx(g,s) + u.dg(g,s)*dgdx(g,s) + u.ds(g,s)*dsdx(g,s)
def dudz(g,s):
    return u.dx(g,s,0)*dxdz(g,s) + u.dg(g,s)*dgdz(g,s) + u.ds(g,s)*dsdz(g,s)

def dphidx(g,s):
    return phi.dx(g,s,0)*dxdx(g,s) + phi.dg(g,s)*dgdx(g,s) + phi.ds(g,s)*dsdx(g,s)
def dphidz(g,s):
    return phi.dx(g,s,0)*dxdz(g,s) + phi.dg(g,s)*dgdz(g,s) + phi.ds(g,s)*dsdz(g,s)

### Below we define the various terms of the FO equations

# Ice viscosity
def eta_v(g,s):
    return df.Constant(b)/2.*(dudx(g,s)**2 \
                +0.25*dudz(g,s)**2 \
                + eps_reg)**((1.-n)/(2*n))

# Longitudinal stress
def membrane_xx(g,s):
    return dphidx(g,s)*eta_v(g,s)*4*dudx(g,s)

# Vertical shear stress
def shear_xz(g,s):
    return dphidz(g,s)*eta_v(g,s)*dudz(g,s)

# Driving stress
def tau_dx(g,s):
    return 1./Lmid*rho*g*S.dx(0)*phi(g,s)

# Create a vertical integrator using gauss-legendre quadrature
points = np.array([[0.    , 0.    ],
       [0.    , 0.4688],
       [0.    , 0.8302],
       [0.    , 1.    ],
       [0.4688, 0.    ],
       [0.4688, 0.4688],
       [0.4688, 0.8302],
       [0.4688, 1.    ],
       [0.8302, 0.    ],
       [0.8302, 0.4688],
       [0.8302, 0.8302],
       [0.8302, 1.    ],
       [1.    , 0.    ],
       [1.    , 0.4688],
       [1.    , 0.8302],
       [1.    , 1.    ]])

weights = np.array([0.05943844, 0.10524846, 0.06748384, 0.01160488, 0.10524846,
       0.18636489, 0.11949456, 0.02054892, 0.06748384, 0.11949456,
       0.07661824, 0.01317568, 0.01160488, 0.02054892, 0.01317568,
       0.00226576])

fi = FlowlineIntegrator(points,weights)

P_0 = rho*g*H_c         # Overburden pressure
P_w = rho_w*g*(l-B)     # Water pressure
N = P_0 - P_w           # Effective pressure

# Basal Shear stress (linear case)
tau_b = beta2*N*u(0,1)

# Residual of the first order equation
R_stress = (- fi.intz(membrane_xx)*H_c*Lmid*width/2. - fi.intz(shear_xz)*H_c*Lmid*width/2. - phi(0,1)*tau_b*width/2.*Lmid - phi(1,0.8)*alpha2*u(1,0.8)*Lmid*H_c - fi.intz(tau_dx)*H_c*width/2.*Lmid)*df.dx 

# The hydrostatic boundary condition at the terminus
R_stress += 1./2*(P_0*H_c - P_w*(l-B) - df.Constant(backstress)*H_c)*nhat[0]*phibar*df.ds(1)

####################################################################
###############  Projection from DG to CG thickness  ###############
####################################################################

# It is useful to have a CG1 approximation for taking thickness derivatives
# which is (sort of) needed for the momentum balance.  Here we project
# the midpoint DG0 thickness onto the CG1 space.  This is tantamount 
# to computing derivatives of the cell centered thickness using centered
# differences
R_c = (H_c-Hmid)*xsi_c*df.dx

####################################################################
##########################  MASS BALANCE  ##########################
####################################################################

# Solve the transport equation using DG0 finite elements which are 
# first order accurate but unconditionally TVD and positivity-preserving,
# and also conserves mass perfectly.

# Grid velocity
v = dLdt*x_spatial[0]   

# Inter element flux (upwind)
uH = df.avg((ubar - v))*df.avg(Hmid*width) + 0.5*abs(df.avg(width*(ubar - v)))*df.jump(Hmid*width)

# Residual of the transport equation with a zero-flux upstream boundary.
R_mass = (Lmid*width*dHdt*xsi + Lmid*Hmid*dWdt*xsi + Hmid*width*dLdt*xsi - xsi.dx(0)*(ubar-v)*width*Hmid - Lmid*width*adot*xsi)*df.dx + uH*df.jump(xsi)*df.dS + (U[0] - v)*Hmid*width*xsi*df.ds(1)

####################################################################
#########################  LENGTH MODEL  ###########################
####################################################################
# Calving thickness (when thickness is less than this, calving begins)
# This is imposed strongly when using the local length evolution equation 
H_calving = (1+df.Constant(q))*D*rho_w/rho

# Calving velocity proportional to the degree to which the calving thickness
# exceeds the thickness.  As k_calving -> infty, this becomes a 
# Dirichlet condition
U_calving = df.Constant(k_calving)*softplus(H_calving-H,0,alpha=0.5)**2

# GLOBAL - derived from integrating the mass conservation equation across 
# the model domain then using the boundary condition u_c = u_t - dL/dt.
# Very stable, works good, not clear how to generalize to multidimensional case
R_len = (dLdt*width*H + L*dWdt*H + L*width*(dHdt - adot))*chi*df.dx + U_calving*width*H*chi*df.ds(1)

# LOCAL - derived from a total differentiation of the calving condition
# Works well as long as dHdx>0.  Otherwise undefined and blows up.  This
# specifically happens during retreats across a basin.
#R_len = (dHdt - 1./L*H_calving.dx(0)*dLdt)*chi*df.ds(1)

####################################################################
#########################  TOTAL RESIDUAL  #########################
####################################################################

# We solve all these simulataneously.  Add the residuals together and 
# differentiate for the complete system.
R = R_stress + R_c + R_mass + R_len
J = df.derivative(R,U,dU)

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

# Define variational problem subject to no Dirichlet BCs, but with a 
# thickness bound, plus form compiler parameters for efficiency.
mass_problem = df.NonlinearVariationalProblem(R,U,bcs=[],J=J,form_compiler_parameters=ffc_options)
mass_problem.set_bounds(l_bound,u_bound)

# Create an instance of vinewtonrsls, PETSc's variational inequality
# solver.  Not clear that this is needed, given that the ice should never be 
# negative by design.
mass_solver = df.NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'
mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = True
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
mass_solver.parameters['snes_solver']['maximum_iterations'] = 10
mass_solver.parameters['snes_solver']['report'] = False

#####################################################################
############### INITIALIZE PLOTTING AND STORAGE #####################
#####################################################################

# Basic animation utilities
if PLOT:
    plt.ion()

    fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(7,7))
    x = mesh.coordinates()*L_initial
    surface = df.project(S)
    base = df.project(B)
    bed = df.project(Bhat)

    ph_base, = ax[0].plot(x,base.compute_vertex_values(),'b-')
    ph_bed, = ax[0].plot(x,bed.compute_vertex_values(),'k:')
    ph_surface, = ax[0].plot(x,surface.compute_vertex_values(),'b-')
    ph_sealevel, = ax[0].plot([0,L_initial],[0,0],'k:')
    
    us = df.project(u(0,0))
    ub = df.project(u(0,1))
    ph_us, = ax[1].plot(x,us.compute_vertex_values())
    ph_ub, = ax[1].plot(x,ub.compute_vertex_values())
    
    ax[0].set_ylim(-1000,2500)
    ax[1].set_ylim(0,400)
    ax[1].set_xlim(0,L_initial)

    plt.pause(0.00001)

# Objects to store various quantities 
if STOREALL:
    ubar_list = []  # depth-average velocity sols
    udef_list = []  # deformational velocity sols
    H_list = []     # thickness sols

vol_list = []       # ice volume
Hmax_list = []      # max thickness
Hterm_list = []     # terminus thickness
Hcalv_list = []     # calving thickness
L_list = []         # glacier length
t_list = []         # time

# Perform the time integration
t = t_start
while t<t_end:
    dt.assign(dt_float)
    
    # HERE YOU CAN ADD SOME CODE TO MESS WITH THE SURFACE MASS BALANCE TO
    # TEST ADVANCE AND RETREAT
    if t>10:
        adot.vector()[:] += 1.0
    if t>2000:
        adot.vector()[:] -= 1.8

    # Solve the equations.  If they fail, try again with the iteration ICs reset
    # to something basic.  Will keep on trucking if unconverged, so be careful or
    # comment 536 and 539 to trigger an error on nonconvergence
    # Compute time step as weighted average of max and previous time step
    dt_float = theta_dt*dt_max + (1-theta_dt)*dt_float
    dt.assign(dt_float)
    converged = False
    
    # Iterate over progressively smaller time steps until convergence
    while not converged:
        # Solve with current time step size
        mass_solver = df.NonlinearVariationalSolver(mass_problem)
        mass_solver.parameters['nonlinear_solver'] = 'snes'
        mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
        mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
        mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
        mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
        #mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
        mass_solver.parameters['snes_solver']['maximum_iterations'] = 10
        mass_solver.parameters['snes_solver']['report'] = False
        p = mass_solver.solve()
        converged = bool(p[1])
        if not converged:
            # If convergence fails, halve the time step and reset initial velocity guesses
            dt_float = dt_float/2.
            dt.assign(dt_float)
            assigner.assign(U,[ze,ze,ze,H0_c,H0,L0])

    # Update previous solutions for time finite differences
    assigner_inv.assign([un,u2n,u3n,H0_c,H0,L0],U)

    # Reinterpolate the bed and mass balance
    Bhat.interpolate(Bed(L0(0),degree=1,element=E_Q))
    adot.interpolate(Adot(L0(0),degree=1,element=E_Q))

    # Update the animation
    if PLOT:
        if round(t)%PLOT_INTERVAL<0.01:
            surface = df.project(S)
            base = df.project(B)
            bed = df.project(Bhat)
            us = df.project(u(0,0))
            ub = df.project(u(0,1))
            
            ax[0].set_xlim(0,L0(0))
            ax[1].set_xlim(0,L0(0))
         
            x = mesh.coordinates()*L0(0.1)

            ph_base.set_xdata(x)
            ph_base.set_ydata(base.compute_vertex_values())
      
            ph_bed.set_xdata(x)
            ph_bed.set_ydata(bed.compute_vertex_values())

            ph_surface.set_xdata(x)
            ph_surface.set_ydata(surface.compute_vertex_values())

            ph_us.set_xdata(x)
            ph_us.set_ydata(us.compute_vertex_values())
            
            ph_ub.set_xdata(x)
            ph_ub.set_ydata(ub.compute_vertex_values())

            ph_sealevel.set_xdata(x)
            ph_sealevel.set_ydata(np.zeros_like(x))
            
            plt.pause(0.00001)
    
    # Store solutions
    if STOREALL:
        ubar_list.append(un.vector().get_local())
        udef_list.append(u2n.vector().get_local())
        H_list.append(H0.vector().get_local())
        
    vol_list.append(df.assemble(width*H*L*df.dx)) # Compute total volume
    Hmax_list.append(H0.vector().max())
    Hterm_list.append(H0_c(1))
    Hcalv_list.append(df.assemble(H_calving*df.ds(1)))
    L_list.append(L0(0.1))
    t_list.append(t)

    # Prints the time, the max ice thickness, ice volume, length
    # terminus thickness, and calving thickness
    print (t,Hmax_list[-1],vol_list[-1],L_list[-1],Hterm_list[-1],Hcalv_list[-1])
    t+=dt_float








