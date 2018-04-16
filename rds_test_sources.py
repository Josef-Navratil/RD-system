#Solution to Schnackenberg problem with Dirichlet boundary conditions
import random
from dolfin import *
import shutil #package to remove old results directory

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


'''
#Class for boundary conditions
center = Point(0.0, 0.0)
radius = 0.05
class Cylinder(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        return(on_boundary and (r < 2*radius + sqrt(DOLFIN_EPS)))
    def snap(self, x):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        if r <= radius:
            x[0] = center[0] + (radius / r)*(x[0] - center[0])
            x[1] = center[1] + (radius / r)*(x[1] - center[1])

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and ((x[1] < 0.0+DOLFIN_EPS) or (x[1] > (10.0 - DOLFIN_EPS))))

class Walls2(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and ((x[0] < 0.0+DOLFIN_EPS) or (x[0] > (15.0 - DOLFIN_EPS))))

'''


#shutil.rmtree('/results') #remove old results directory



# Model parameters
dt     = 0.1  # time step
theta  = 0.5      # time stepping family; theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
#mesh = UnitSquareMesh(60, 60)
mesh = RectangleMesh(Point(0.0,0.0), Point(15.0, 10.0), 100, 100, "right/left")
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
v2 = Function(VE) #one component function
v3 = Function(VE) #one component function

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# Define parameters
lmbda=0.1
a=0.1
b=0.85
baa=(-1+2*b/(a+b))
bab=(a+b)*(a+b)
bba=-2*b/(a+b)
bbb=-(a+b)*(a+b)
#d1=0.02
#d2=0.3
d1=0.2
d2=5


# Define source functions

f_1 = Expression('pow(x[0]-5.0,2)+pow(x[1]-5.0,2)<1 ? 2 : 0',
                 degree=1)

f_2 = Expression('exp(-pow(x[0]-5.0,2)-pow(x[1]-5.0,2))-0.135', degree=1)
f_3 = Expression('exp(-pow(x[0]-3.0,2)-pow(x[1]-3.0,2))-0.135', degree=1)
# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
v2.interpolate(f_1)
v3.interpolate(f_2)

# Verify Turing instability conditions
#if ((baa*bbb-bab*bba)<0):
#	print("Det B is negative!")
#	break
#if ((baa+bbb)>0):
#	print("Tr B is positive!")
#	break
#if ((baa<0) or (bbb>0) or (bba*bab>0)):
#	print("Wrong coefficients signs")
#	break

# Split mixed functions
dv, dw = split(du)
v,  w  = split(u)
v0, w0 = split(u0)

# Weak statement of the equations 
L0 = d1*dt*inner(grad(v0), grad(p))*dx - dt*baa*v0*p*dx - dt*bab*w0*p*dx - 2*(a+b)*dt*v0*w0*p*dx - (b/(a+b)*(a+b))*dt*v0*v0*p*dx - dt*v0*v0*w0*p*dx
L1 = d1*dt*inner(grad(v), grad(p))*dx - dt*baa*v*p*dx - dt*bab*w*p*dx - 2*(a+b)*dt*v*w*p*dx- (b/(a+b)*(a+b))*dt*v*v*p*dx - dt*v*v*w*p*dx
M0 = d2*dt*inner(grad(w0), grad(q))*dx - dt*bba*v0*q*dx - dt*bbb*w0*q*dx + 2*(a+b)*dt*v0*w0*q*dx+(b/(a+b)*(a+b))*dt*v0*v0*q*dx +dt*v0*v0*w0*q*dx +f_2*0.5*dt*(1-sign(w0))*w0*q*dx-f_3*dt*0.5*(1+sign(w0))*w0*q*dx
M1 = d2*dt*inner(grad(w), grad(q))*dx - dt*bba*v*q*dx - dt*bbb*w*q*dx + 2*(a+b)*dt*v*w*q*dx+ (b/(a+b)*(a+b))*dt*v*v*q*dx + dt*v*v*w*q*dx +f_2*0.5*dt*w*(1-sign(w))*w*q*dx-f_3*dt*0.5*(1+sign(w))*w*q*dx
L =  v*p*dx - v0*p*dx+ w*q*dx - w0*q*dx + theta*L1 + (1-theta)*L0 + theta*M1 + (1-theta)*M0

# Compute directional derivative about u in the direction of du (Jacobian)
J = derivative(L, u, du)
a = derivative(L, u, du)


#boundary conditions
Dirchlt = Constant((0,0))


def boundary(x, on_boundary):
    return on_boundary
bcs = DirichletBC(ME, Dirchlt, boundary)


# Create nonlinear problem and Newton solver
#problem = RDS(a, L)
#solver = NewtonSolver()
#solver.parameters["linear_solver"] = "lu"
#solver.parameters["convergence_criterion"] = "incremental"
#solver.parameters["relative_tolerance"] = 1e-6

# Output file
ufile = File("results/output.pvd", "compressed")
#vfile = File("results/output2.pvd", "compressed")

problem = NonlinearVariationalProblem(L, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 10


# Step in time
t = dt
T = 2000*dt
k=0
while (t < T):
    print("t=%f, iteration=%i" %(t,k))
    solver.solve()
    end()
    if k%4==0:
        ufile << (u0.split()[0], t)
    k+=1
#   vfile << (u0.split()[1], t)
    u0.assign(u)
    t += dt
if k%4!=0:
    file << (u.split()[0], t)
    
#while (t < T):
#    print("t=%f", t)
#    t += dt
#    u0.vector()[:] = u.vector()
#    file << (u.split()[0], t)
#    solver.solve(problem, u.vector())
#    u
#    if k%4==0:
#	    file << (u.split()[0], t)
#    k+=1
#if k%4!=0:
#    file << (u.split()[0], t)
#plot(u.split()[0])
#interactive()
