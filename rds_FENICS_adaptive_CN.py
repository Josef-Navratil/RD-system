#Solution of RDS problem with sources
import random
from dolfin import *
import shutil #package to remove old results directory
from sys import argv
import time
from sys import exit
# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self, **kwargs):
        random.seed(0.1 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 0.2*(random.random())
        values[1] = 0.2*(random.random())
    def value_shape(self):
        return (2,)

# Class for interfacing with the Newton solver
class RDS(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)




# Model parameters
dt     = Constant(0.1)  # time step
theta  = 0.5      # time stepping family; theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
mesh = RectangleMesh(Point(0.0,0.0), Point(15.0, 10.0), 100, 80, "right/left")
#mesh = UnitSquareMesh(60, 60)
#mesh=refine(Mesh("mesh.xml"))
V=FiniteElement("Lagrange", mesh.ufl_cell(),1)
ME = FunctionSpace(mesh, V*V)
VE = FunctionSpace(mesh, V)
# Define trial and test functions
du    = TrialFunction(ME)
q, p  = TestFunctions(ME)


# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step
u_large = Function(ME)
u_small = Function(ME)
u_old = Function(ME)
u0_old = Function(ME)
v2 = Function(VE) #one component function
v3 = Function(VE) #one component function


# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)



# Load parameters from console
d1=float(argv[1])
d2=float(argv[2])
T=float(argv[3])
BC=argv[4]
Kin=argv[5]
n_iter=int(argv[6])


# Define source functions
'''
f_1 = Expression('pow(x[0]-11.0,2)+pow(x[1]-6.0,2)<2.25 ? 1.0 : 0',
                 degree=1)
f_10 = Expression('pow(x[0]-3.0,2)+pow(x[1]-3.0,2)<2.25 ? 1.0 : 0',
                 degree=1)
f_2 = Expression('1.0*(exp(-pow(x[0]-11.0,2)-pow(x[1]-6.0,2))-0.1054)', degree=1)
f_3 = Expression('1.0*(exp(-pow(x[0]-3.0,2)-pow(x[1]-3.0,2))-0.1054)', degree=1)
'''

# Milder sources
f_1 = Expression('pow(x[0]-11.0,2)+pow(x[1]-6.0,2)<16 ? 1.0 : 0',
                 degree=1)
f_10 = Expression('pow(x[0]-3.0,2)+pow(x[1]-3.0,2)<16 ? 1.0 : 0',
                 degree=1)

f_2 = Expression('1.0*(exp(-0.25*pow(x[0]-11.0,2)-0.25*pow(x[1]-6.0,2))-0.0111)', degree=1)
f_3 = Expression('1.0*(exp(-0.25*pow(x[0]-3.0,2)-0.25*pow(x[1]-3.0,2))-0.0111)', degree=1)



# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
v2.interpolate(f_1)
v3.interpolate(f_2)



# Split mixed functions
dv, dw = split(du)
v,  w  = split(u)
v0, w0 = split(u0)

# Weak statement of the equations 
# Schnackenberg kinetics
if Kin=='SCH':
    # Define parameters
    a=0.1
    b=0.85
    baa=(-1+2*b/(a+b))
    bab=(a+b)*(a+b)
    bba=-2*b/(a+b)
    bbb=-(a+b)*(a+b)
    
    L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*baa*v0*p*dx - dt*bab*w0*p*dx - 2*(a+b)*dt*v0*w0*p*dx - (b/(a+b)*(a+b))*dt*v0*v0*p*dx - dt*v0*v0*w0*p*dx
    L1 = d1*dt*inner(grad(v), grad(p))*dx - dt*baa*v*p*dx - dt*bab*w*p*dx - 2*(a+b)*dt*v*w*p*dx- (b/(a+b)*(a+b))*dt*v*v*p*dx - dt*v*v*w*p*dx
    M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*bba*v0*q*dx - dt*bbb*w0*q*dx + 2*(a+b)*dt*v0*w0*q*dx+(b/(a+b)*(a+b))*dt*v0*v0*q*dx +dt*v0*v0*w0*q*dx +f_1*f_2*0.5*dt*(1-sign(w0))*w0*q*dx+f_10*f_3*0.5*dt*(1+sign(w0))*w0*q*dx
    M1 = d2*dt*inner(grad(w), grad(q))*dx - dt*bba*v*q*dx - dt*bbb*w*q*dx + 2*(a+b)*dt*v*w*q*dx+ (b/(a+b)*(a+b))*dt*v*v*q*dx + dt*v*v*w*q*dx +f_1*f_2*0.5*dt*(1-sign(w))*w*q*dx+f_10*f_3*0.5*dt*(1+sign(w))*w*q*dx
    L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0
# FitzHugh-Nagumo Kinetics
elif Kin=='F':
    # Define parameters
    baa=0.6
    bab=-4.5
    bba=1.5
    bbb=-2
    
    # Variational form of the problem
    L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*baa*v0*p*dx - dt*bab*w0*p*dx 
    L1 = d1*dt*inner(grad(v), grad(p))*dx - dt*baa*v*p*dx - dt*bab*w*p*dx
    M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*bba*v0*q*dx - dt*bbb*w0*q*dx+ 2*dt*v0*v0*p*dx + dt*v0*v0*v0*p*dx  +f_1*f_2*0.5*dt*(1-sign(w0))*w0*q*dx+f_10*f_3*0.5*dt*(1+sign(w0))*w0*q*dx
    M1 = d2*dt*inner(grad(w), grad(q))*dx - dt*bba*v*q*dx - dt*bbb*w*q*dx + 2*dt*v*v*p*dx + dt*v*v*v*p*dx  +f_1*f_2*0.5*dt*(1-sign(w))*w*q*dx+f_10*f_3*0.5*dt*(1+sign(w))*w*q*dx
    L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0

# Thomas kinetics
elif Kin=='T':
    # Define parameters
    baa=6
    bab=-45
    bba=15
    bbb=-20
    
    # Variational form of the problem    
    L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*baa*v0*p*dx - dt*bab*w0*p*dx + rd*dt*v0*w0*p*dx + baa*rt*dt*v0*w0*w0*p*dx
    L1 = d1*dt*inner(grad(v), grad(p))*dx - dt*baa*v*p*dx - dt*bab*w*p*dx + rd*dt*v*w*p*dx + baa*rt*dt*v*w*w*p*dx
    M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*bba*v0*q*dx - dt*bbb*w0*q*dx - rd*dt*v0*w0*q*dx - baa*rt*dt*v0*w0*w0*q*dx + f_1*f_2*0.5*dt*(1-sign(w0))*w0*q*dx + f_10*f_3*0.5*dt*(1+sign(w0))*w0*q*dx
    M1 = d2*dt*inner(grad(w), grad(q))*dx - dt*bba*v*q*dx - dt*bbb*w*q*dx - rd*dt*v*w*q*dx - baa*rt*dt*v*w*w*q*dx + f_1*f_2*0.5*dt*(1-sign(w))*w*q*dx + f_10*f_3*0.5*dt*(1+sign(w))*w*q*dx
    L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0

# Liu-Liaw-Maini kinetics
elif Kin=='LLM':
    # Define parameters
    baa=0.899
    bab=1
    bba=-0.899
    bbb=-0.91
    rd=2
    rt=3.5
    
    # Variational form of the problem   
    L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*baa*v0*p*dx - dt*bab*w0*p*dx + rd*dt*v0*w0*p*dx + baa*rt*dt*v0*w0*w0*p*dx
    L1 = d1*dt*inner(grad(v), grad(p))*dx - dt*baa*v*p*dx - dt*bab*w*p*dx + rd*dt*v*w*p*dx + baa*rt*dt*v*w*w*p*dx
    M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*bba*v0*q*dx - dt*bbb*w0*q*dx - rd*dt*v0*w0*q*dx - baa*rt*dt*v0*w0*w0*q*dx+ f_1*f_2*0.5*dt*(1-sign(w0))*w0*q*dx + f_10*f_3*0.5*dt*(1+sign(w0))*w0*q*dx
    M1 = d2*dt*inner(grad(w), grad(q))*dx - dt*bba*v*q*dx - dt*bbb*w*q*dx - rd*dt*v*w*q*dx - baa*rt*dt*v*w*w*q*dx + f_1*f_2*0.5*dt*(1-sign(w))*w*q*dx + f_10*f_3*0.5*dt*(1+sign(w))*w*q*dx
    L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0
 
 
else:
    print("Wrong switch for kinetics, break")
    exit()



# Verify Turing instability conditions
if ((baa*bbb-bab*bba)<0):
	print("Det B is negative!")

if ((baa+bbb)>0):
	print("Tr B is positive!")

if ((baa<0) or (bbb>0) or (bba*bab>0)):
	print("Wrong coefficients signs")
	


# Compute directional derivative about u in the direction of du (Jacobian)
J = derivative(L, u, du)
a = derivative(L, u, du)


# Boundary conditions
# Dirichlet boundary conditions
if BC=='D':
    Dirchlt = Constant((0,0))
    def boundary(x, on_boundary):
        return on_boundary
    bcs = DirichletBC(ME, Dirchlt, boundary)

# Neumann boundary conditions
elif BC=='N':
    bcs=[]

else:
    print("Wrong choice of boundary conditions")
    exit()

# Create nonlinear problem and Newton solver
#problem = RDS(a, L)
#solver = NewtonSolver()
#solver.parameters["linear_solver"] = "lu"
#solver.parameters["convergence_criterion"] = "incremental"
#solver.parameters["relative_tolerance"] = 1e-6

# Output file
file_location="resultsSO-%.3f-%.3f-%s-%s"%(d1,d2,Kin,BC)+"/output.pvd"
file_location2="resultsSO-%.3f-%.3f-%s-%s"%(d1,d2,Kin,BC)+"/output2.pvd"
ufile = File(file_location, "compressed")
vfile = File(file_location2, "compressed")

problem = NonlinearVariationalProblem(L, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 20


# Step in time
t = dt
k = 0
u1_vec = u0.vector()

error_file = open('error.txt', 'w', buffering = 1)


m = 3
m_0 = m
dt = 0.5

dt_old = dt
t_old = t
TOL = 1e-3

while (k < 10):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt


while k < 8000:
    print("t=%f, iteration=%i, step=%f" %(t,k,dt))
    dt_large = m_0*dt
    u_old.assign(u)
    u0_old.assign(u0)
    dt_old = dt
    t_old = t
    dt = dt_large
    print(t, dt, dt_large)
    (no_of_iterations,converged) = solver.solve()
    u_large.assign(u)
    u.assign(u_old)
    t = t_old
    dt = dt_old
    for k in range (0, m_0):
        (no_of_iterations,converged) = solver.solve()
        u0.assign(u)
        t += dt
    print(t)
    err = norm(u_large.vector()-u.vector())
    dt = TOL*dt*dt*(m_0*m_0-1)/err
    print(dt,m_0,err)
    if dt<(dt_old/m_0) and m_0>2:
        m_0 = m_0-1
        u.assign(u_old)
        t = t_old
        u0.assign(u0_old)
        dt = dt_old
        print("assigned1")
        time.sleep(2.0)
    elif dt<(dt_old/m_0) and m_0 == 2:
        dt = dt_old/2
        u.assign(u_old)
        t = t_old
        u0.assign(u0_old)
        print("assigned2")
        time.sleep(2.0)
    else:
        m_0 = m
        u_large.assign(u)
    k+=1
    
    u0.assign(u_old)
    t += dt

        
error_file.close()
