#Solution of RDS problem with sources
from dolfin import *
import shutil #package to remove old results directory
from sys import argv
from sys import exit
from os import remove
from initial_cond import InitialConditions 





def solve_pde(d1, d2, TOL, dT, source,  BC): 
    # Model parameters
    dt = Constant(dT)
    theta  = 0.5      # time stepping family; theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
    Kin = 'SCH'

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
    v2 = Function(VE) #one component function
    v3 = Function(VE) #one component function


    # Create intial conditions and interpolate
    u_init = InitialConditions(degree=1)
    u.interpolate(u_init)
    u0.interpolate(u_init)

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

    f_2 = Expression('{0}*(exp(-0.25*pow(x[0]-11.0,2)-0.25*pow(x[1]-6.0,2))-0.0111)'.format(source), degree=1)
    f_3 = Expression('{0}*(exp(-0.25*pow(x[0]-3.0,2)-0.25*pow(x[1]-3.0,2))-0.0111)'.format(source), degree=1)



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
    folder_path = "results/results-%.3f-%.8f-%s-%s-%.1f/"%(d1,d2,Kin,BC, source) 
    file_location_v = folder_path + "output.pvd"
    file_location_u = folder_path + "output2.pvd"
    ufile = File(file_location_u, "compressed")
    vfile = File(file_location_v, "compressed")

    problem = NonlinearVariationalProblem(L, u, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['maximum_iterations'] = 20


    # Step in time
    t = float(dt)
    dt_change = 1.5
    dT = float(dt)
    k = 0
    u1_vec = u0.vector()
    no_of_files = 0
    file_path = folder_path + "error.txt"
    error_file = open(file_path, 'w', buffering = 1)
    err = 1

    while True:
        print("t=%f, iteration=%i, step=%f" %(t,k,dt))
        try:
            (no_of_iterations,converged) = solver.solve()
        except:
            quit()
        end()
        k+=1
        if converged:
            if float(dt) > 1e6 or k > 100000 or err < TOL:
                break
            if k%10==0:
                u_vec = u.vector()
                err = norm(u_vec-u1_vec)/norm(u_vec)
                print(err)
                error_file.write(str(err)+'\n')
                
                if err < 1e-2:
                    dt.assign(Constant(1.5*dT))
                    dT = float(dt)
                    
                if err > 1.0:
                    dt.assign(Constant(0.5*dT))
                    dT = float(dt)
                
                ''' Uncomment for saving last 1000 results
                if k%20==0:
                    ufile << (u0.split()[1], t)
                    no_of_files += 1
                    if no_of_files >= 100:
                        count = no_of_files - 100
                        file_location = folder_path + "output%06d.vtu"%(no_of_files-100)
                        print(file_location)
                        remove(file_location)
                '''
                u1_vec = u0.vector()
            u0.assign(u)
            t = t + dT
        else:
            dt = 0.25*dt
        if k==10:
            break
            
    ufile << (u0.split()[0], t)
    vfile << (u0.split()[1], t)
    File(folder_path + 'saved_mesh.xml') << mesh
    File(folder_path + 'saved_v0.xml') << u0
    File(folder_path + 'saved_v.xml') << u

    norm_h1 = norm(u,'H1')
    error_file.write('norm: ' + str(norm_h1))
    error_file.close()
    return norm_h1
