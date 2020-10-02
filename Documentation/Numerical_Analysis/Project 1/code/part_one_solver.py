import numpy as np
import scipy.signal as signal

def rhs(x, y):
    # Element-wise multiplication (9b)
    return np.multiply(x, (x - y)**3)

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def problem_one_solver(rightHandSide, *args, **kwargs):

    differential_operator = np.dot(4, np.eye(7))
    differential_operator[0,1] = -1
    differential_operator[0,4] = -1
    differential_operator[1,2] = -1
    differential_operator[1,5] = -1
    differential_operator[2,3] = -1
    differential_operator[4,5] = -1
    differential_operator[4,6] = -1
    differential_operator[1,0] = -1
    differential_operator[2,1] = -1
    differential_operator[3,2] = -1
    differential_operator[5,4] = -1
    differential_operator[6,4] = -1
    # X and Y Coordinates for evaluating right hand side
    X = np.concatenate([[1],[2],[3],[4],[1],[2],[1]], axis = 0)
    Y = np.concatenate([[1],[1],[1],[1],[2],[2],[3]], axis = 0)
    # Actually evaluate the rhs at points
    rhs_return = rightHandSide(X, Y)

    # Solve for solution vector
    solution_vector = np.linalg.solve(differential_operator, rhs_return)

    # define the mesh grid for embedding solution and for visualization
    X, Y = np.meshgrid(np.arange(0,float(6)), np.arange(0,float(5)))
    
    U = np.vstack([[0,0,0,0,0,0],
        [0, solution_vector[0],
        solution_vector[1],
        solution_vector[2],
        solution_vector[3], 0],
        [0,solution_vector[4],
         solution_vector[5],0,0,0],
        [0,solution_vector[6],0,0,0,0],
        [0,0,0,0,0,0]])

    # Remove x and y coordinates that are not within the domain or boundary
    arr_temp = np.vstack([[1,1,1],[1,0,1],[1,1,1]])
    convolve2d_temp = np.logical_not(signal.convolve2d(U, arr_temp, mode='same'))
    zero_indices = np.where(convolve2d_temp)
   
    X[zero_indices] = np.nan
    Y[zero_indices] = np.nan   

    return X, Y, U

if __name__ == "__main__":

    X, Y, Z = problem_one_solver(rhs)