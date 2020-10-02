import numpy as np
import scipy.signal as signal

def rhs(x, y):
    # Element-wise multiplication
    return np.multiply(x, (x - y)**3)

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def problem_one_part_two_solver(rhs_func, *args, **kwargs):
    
    # Differential operator
    differential_operator = np.dot(4, np.eye(43))
    row_indices = np.concatenate([[1],[1],[2],[2],[2],[3],[3],[3],[4],[4],[4],
        [5],[5],[5],[6],[6],[6],[7],[7],[8],[8],[8],[9],[9],[9],[9],
        [10],[10],[10],[10],[11],[11],[11],[11],[12],[12],[12],[12],
        [13],[13],[13],[13],[14],[14],[14],[15],[15],[15],[16],[16],
        [16],[16],[17],[17],[17],[17],[18],[18],[18],[18],[19],[19],
        [19],[19],[20],[20],[20],[21],[21],[22],[22],[22],[23],[23],
        [23],[23],[24],[24],[24],[24],[25],[25],[25],[25],[26],[26],
        [26],[27],[27],[27],[28],[28],[28],[28],[29],[29],[29],[29],
        [30],[30],[30],[31],[31],[32],[32],[32],[33],[33],[33],[33],
        [34],[34],[34],[35],[35],[35],[36],[36],[36],[36],[37],[37],
        [37],[38],[38],[38],[39],[39],[39],[39],[40],[40],[40],[41],
        [41],[42],[42],[42],[43],[43]], axis = 0)

    column_indices = np.concatenate([[2],[8],[1],[9],[3],[2],[10],[4],[3],[11],[5],[4],
        [12],[6],[5],[13],[7],[6],[14],[1],[9],[15],[2],[10],[16],[8],[3],
        [11],[17],[9],[4],[12],[18],[10],[5],[13],[19],[11],[6],[14],[20],
        [12],[7],[21],[13],[8],[16],[22],[9],[17],[23],[15],[10],[18],[24],
        [16],[11],[19],[25],[17],[12],[20],[26],[18],[19],[13],[21],[20],[14],
        [15],[23],[27],[16],[24],[28],[22],[17],[25],[29],[23],[18],[26],[30],
        [24],[19],[31],[25],[22],[28],[32],[23],[29],[33],[27],[24],[30],[34],
        [28],[25],[31],[29],[26],[30],[27],[33],[35],[28],[34],[36],[32],[29],
        [37],[33],[32],[36],[38],[33],[37],[39],[35],[34],[40],[36],[35],[39],
        [41],[36],[40],[42],[38],[37],[43],[39],[38],[42],[41],[39],[43],[42],
        [40]], axis = 0)


    for k in range(0, 43):
        differential_operator[row_indices[k], column_indices[k]]= -1

    arbitrary_domain = np.vstack([
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0]
    ])
#     # Retrieve x and y points that will be evaluated
    y_interior, x_interior = np.nonzero(arbitrary_domain)
    x_interior = np.dot(0.5, x_interior)
    y_interior = np.dot(0.5, y_interior)
#   # Evaluate right hand side at interior points
    return_vector = np.dot(0.5 ** 2, rhs_func(x_interior, y_interior))
#   # Solve the solution vector
    solution_vector = np.linalg.solve(differential_operator, return_vector)
#   # X and Y coordinates
    X, Y = np.meshgrid(np.arange(0, 4.5, 0.5), np.arange(0, 5.5, 0.5))

    seven_by_one = np.zeros((7, 1))
    one_by_eleven = np.zeros((1, 11))

    U = np.concatenate((seven_by_one, arbitrary_domain, seven_by_one), axis  = 1)
    U = np.concatenate((one_by_eleven, U, one_by_eleven), axis = 0).T
    
    (i, j) = (U != 0).nonzero()
    for solution in np.arange(0, np.size(solution_vector)).reshape(-1):
      U[i, j] = solution_vector[i]

    # spy for debugging
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spy(U);
    
   # Remove x and y coordinates that are not within the arbitrary_domain or boundary
    arr_temp = np.vstack([[1,1,1],[1,0,1],[1,1,1]])
    convolve2d_temp = np.logical_not(signal.convolve2d(U, arr_temp, mode='same'))
    zero_indices = np.where(convolve2d_temp)

    X[zero_indices] = np.nan
    Y[zero_indices] = np.nan   
    
    return X, Y, U

if __name__ == "__main__":
    X, Y, Z = problem_one_part_two_solver(rhs)