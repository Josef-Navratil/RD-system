from make_computation import solve_pde
import os

def find_bif_point(d1, d2_max, d2_min, BC, TOL, dT, source, TOL_bif):
    k = 0
    if d2_max < d2_min:
        print("upper bound on d2 must be higher than lower bound")
        return False
    d2 = (d2_max + d2_min)/2
    bif_point_error = d2_max - d2_min
    while abs(bif_point_error) > TOL_bif:
        norm = solve_pde(d1, d2, TOL, dT, source, BC)
        if norm < 1e-6:
            d2_min = d2
        else:
            d2_max = d2
        bif_point_error = d2_max - d2_min
        d2 = (d2_min + d2_max)/2
        print(bif_point_error)
        k += 1
        if k == 50:
            return d2
    return d2
    
if __name__ == '__main__':
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_values = open('results/bif_point.txt', 'w')
    BC = 'N'
    TOL = 1e-7
    source = 1.0
    parameter_list = [[0.02, 0.17, 0.08, 0.03, 1e-6], [0.12, 0.90, 0.20, 0.02, 1e-5], [0.5, 3.90, 2.80, 0.02, 1e-4], [1.1, 8.20, 6.20, 0.01, 1e-3], [1.6, 12.90, 10.00, 0.01, 1e-3], [4.0, 33.00, 25.00, 0.01, 1e-3]]
    for [d1, d2_max, d2_min, dT, TOL_bif] in parameter_list:
        d2_bif = find_bif_point(d1, d2_max, d2_min, BC, TOL, dT, source, TOL_bif)
        file_values.write("d1: " + str(d1) + " d2: " + str(d2_bif) + "\n")
        print(d2_bif)
        
    BC = 'D'        
    parameter_list2 = [[0.02, 0.17, 0.08, 0.03, 1e-6], [0.12, 0.90, 0.20, 0.02, 1e-5], [0.5, 3.90, 2.80, 0.02, 1e-4], [1.1, 8.20, 6.20, 0.01, 1e-3], [1.6, 12.90, 10.00, 0.01, 1e-3], [4.0, 38.00, 27.00, 0.01, 1e-3]]
    for [d1, d2_max, d2_min, dT, TOL_bif] in parameter_list:
        d2_bif = find_bif_point(d1, d2_max, d2_min, BC, TOL, dT, source, TOL_bif)
        file_values.write("d1: " + str(d1) + " d2: " + str(d2_bif) + "\n")
        print(d2_bif)
    file_values.close()
    os.system('systemctl poweroff')
    

    
