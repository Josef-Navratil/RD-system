#Solution of RDS problem with sources
import random
from dolfin import *
import shutil #package to remove old results directory
from sys import argv
from sys import exit
# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self, **kwargs):
        random.seed(0.0 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 37.7380 + 0.5*(random.random())
        values[1] = 25.1588 + 0.5*(random.random())
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
dt     = 0.1  # time step
theta  = 0.5      # time stepping family; theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
mesh = RectangleMesh(Point(0.0,0.0), Point(15.0, 10.0), 100, 100, "right/left")
#mesh = UnitSquareMesh(60, 60)
#mesh=refine(Mesh("mesh.xml"))
V = FiniteElement("Lagrange", mesh.ufl_cell(),1)
ME = FunctionSpace(mesh, V*V)
VE = FunctionSpace(mesh, V)
# Define trial and test functions
du    = TrialFunction(ME)
q, p  = TestFunctions(ME)


# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step
v2 = Function(VE) #one component function
v3 = Function(VE) #one component function


# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)



# Load parameters from console
d1=15
d2=800
T=200
BC='N'
Kin='T'
n_iter=1


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
if Kin=='T':
    GA   = 252
    K1   = 0.05
    RRHO = 13
    ALP  = 1.5
    AA   = 150
    BB   = 100
    # Define parameters
    # Variational form of the problem    
    L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*GA*(AA  - v0 - RRHO*v0*w0*pow(1+v0+K1*v0*v0,-1))*p*dx
    L1 = d1*dt*inner(grad(v),  grad(p))*dx - dt*GA*(AA  - v  - RRHO*v*w*pow(1+v+K1*v*v,-1))*p*dx
    M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*(ALP*BB - ALP*w0 - RRHO*v0*w0*pow(1+v0+K1*v0*v0,-1))*q*dx + f_1*f_2*0.5*dt*(1-sign(w0))*w0*q*dx + f_10*f_3*0.5*dt*(1+sign(w0))*w0*q*dx
    M1 = d2*dt*inner(grad(w),  grad(q))*dx - dt*(ALP*BB - ALP*w  - RRHO*v*w*pow(1+v+K1*v*v,-1))*q*dx + f_1*f_2*0.5*dt*(1-sign(w))*w*q*dx + f_10*f_3*0.5*dt*(1+sign(w))*w*q*dx
    L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0
 
else:
    print("Wrong switch for kinetics, break")
    exit()



	


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
file_location="resultsSO-%.2f-%.2f-%s"%(d1,d2,Kin)+"/output.pvd"
file_location2="resultsSO-%.2f-%.2f-%s"%(d1,d2,Kin)+"/output2.pvd"
ufile = File(file_location, "compressed")
vfile = File(file_location2, "compressed")

problem = NonlinearVariationalProblem(L, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 15


# Step in time
t = dt

k=0
while (k < 50):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt
if t>T:
    exit()
dt=0.5
ufile << (u0.split()[1], t)
if t>T:
    sys.exit()
while (k < 1000):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt
dt=1.0
if t>T:
    exit()
ufile << (u0.split()[1], t)
while (k < 1500):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt
    if(k%5==0):
        ufile << (u0.split()[1], t)
dt=3.0
ufile << (u0.split()[1], t)
if t>T:
    exit()
while (k < T):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt
#    if((k%20==0) and (k>5000)):
#        ufile << (u0.split()[1], t)
dt=150
ufile << (u0.split()[1], t)
'''
while (t < 2000000):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    k+=1
    u0.assign(u)
    t += dt
ufile << (u0.split()[1], t)
dt=200
while (t < T):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    if((k%20==0) and (t>3400000)):
        ufile << (u0.split()[1], t)
    k+=1
    u0.assign(u)
    t += dt

print("t=%f, iteration=%i" %(t,k))
solver.solve()
end()
vfile << (u0.split()[0], t)
File('saved_mesh.xml') << mesh
File('saved_v0.xml') << u0
File('saved_v.xml') << u

while (t < T):
   print("t=%f", t)
    t += dt
    u0.vector()[:] = u.vector()
    file << (u.split()[0], t)
    solver.solve(problem, u.vector())
    u
    if k%4==0:
	    file << (u.split()[0], t)
    k+=1
if k%4!=0:
    file << (u.split()[0], t)
plot(u.split()[0])
interactive() 
'''
