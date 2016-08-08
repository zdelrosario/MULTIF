from ad import gh
# {P_0,P_e,A_0,A_e,U_0,U_e,a_0}
dim_mat = [[ 1, 1, 0, 0, 0, 0, 0], # M
           [-1,-1, 2, 2, 1, 1, 1], # L
           [-2,-2, 0, 0,-1,-1,-1]] # T

def fcn(x):
    # P_0 = x[0]
    # P_e = x[1]
    # A_0 = x[2]
    # A_e = x[3]
    # U_0 = x[4]
    # U_e = x[5]
    # a_0 = x[6]

    gam = 1.4; f = 0.05

    return gam*x[2]*x[0]*x[4]**2/x[6]**2*((1+f)*x[5]/x[4]+1) + \
           x[3]*(x[1]-x[0])

grad,_ = gh(fcn)

if __name__ == "__main__":
    # Make nominal values global, len=6
    P_0 = 101e3                     # Freestream pressure, Pa
    P_e = P_0 * 10                  # Exit pressure, Pa
    A_0 = 1.5                       # Capture area, m^2
    A_e = 1.                        # Nozzle exit area, m^2
    U_0 = 100.                      # Freestream velocity, m/s
    U_e = U_0 * 2                   # Exit velocity, m/s
    a_0 = 343.                      # Ambient sonic speed, m/s

    X_nom = [P_0,P_e,A_0,A_e,U_0,U_e,a_0]

    f = fcn(X_nom)
    g = grad(X_nom)
    
