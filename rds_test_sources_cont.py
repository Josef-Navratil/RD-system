#Solution of Schnackenberg problem with Dirichlet boundary conditions
import random
from dolfin import *
import shutil #package to remove old results directory
from sys import argv



# Model parameters
dt     = 1.0  # time step
theta  = 0.5      # time stepping family; theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
#mesh = UnitSquareMesh(60, 60)
mesh = Mesh('saved_mesh.xml')
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
#u_init = InitialConditions(degree=1)
#u.interpolate(u_init)
#u0.interpolate(u_init)
u=Function(ME, 'saved_v.xml')
u0=Function(ME, 'saved_v0.xml')

File('saved_v0.xml') << u0

# Define parameters
d1 = 1.1
d2 = 7.4

# Define source functions
f_1 = Expression('pow(x[0]-11.0,2)+pow(x[1]-6.0,2)<16 ? 1.0 : 0',
                 degree=1)
f_10 = Expression('pow(x[0]-3.0,2)+pow(x[1]-3.0,2)<16 ? 1.0 : 0',
                 degree=1)

f_2 = Expression('1.0*(exp(-0.25*pow(x[0]-11.0,2)-0.25*pow(x[1]-6.0,2))-0.0111)', degree=1)
f_3 = Expression('1.0*(exp(-0.25*pow(x[0]-3.0,2)-0.25*pow(x[1]-3.0,2))-0.0111)', degree=1)
# Create intial conditions and interpolate
v2.interpolate(f_1)
v3.interpolate(f_2)

# Split mixed functions
dv, dw = split(du)
v,  w  = split(u)
v0, w0 = split(u0)

# Weak statement of the equations 
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

# Compute directional derivative about u in the direction of du (Jacobian)
J = derivative(L, u, du)
a = derivative(L, u, du)


#boundary conditions
Dirchlt = Constant((0,0))


def boundary(x, on_boundary):
    return on_boundary
bcs = DirichletBC(ME, Dirchlt, boundary)
#bcs=[]

# Create nonlinear problem and Newton solver
#problem = RDS(a, L)
#solver = NewtonSolver()
#solver.parameters["linear_solver"] = "lu"
#solver.parameters["convergence_criterion"] = "incremental"
#solver.parameters["relative_tolerance"] = 1e-6
# Output file
file_location="resultsSO-SCH-1.1-7.4-D/output.pvd"
ufile = File(file_location, "compressed")
#vfile = File("results/output2.pvd", "compressed")

problem = NonlinearVariationalProblem(L, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['maximum_iterations'] = 10


# Step in time
t = 580000.0
T = 600000.0
dt = 1.0
k = 0
#T=t_end
while (t < T):
    print("t=%f, iteration=%i" %(t,k))
    (iter_nr, coverged) = solver.solve()
    end()
    if iter_nr == 0:
        dt = 2*dt
    if(k%50==0):
        ufile << (u0.split()[1], t)
    k+=1
    u0.assign(u)
    t += dt
print("t=%f, iteration=%i" %(t,k))
solver.solve()
end()
File('saved_mesh.xml') << mesh
File('saved_v0.xml') << u0
File('saved_v.xml') << u
