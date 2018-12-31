from dolfin import Expression
from dolfin import assemble
from dolfin import NonlinearProblem
import random
from dolfin import mpi_comm_world
from dolfin import MPI

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
