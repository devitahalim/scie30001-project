import random

def fake_EM(initial_gaussian_params, max_iterations):
    num_gaussians = len(initial_gaussian_params)
    iteration_params = list() # create an empty list to store the new parameters after each iteration
    for i in range(max_iterations):
        new_params = list()
        for j in range(num_gaussians):
            new_gaussian_params = {
                'mean': initial_gaussian_params[j]['mean']*random.random(),
                'variance':initial_gaussian_params[j]['variance']*random.random()
                }
            new_params.append(new_gaussian_params)
        iteration_params.append(new_params)
    return(iteration_params)


gaussian_params_1 = {'mean': 2, 'variance': 0.5}
gaussian_params_2 = {'mean': 4, 'variance': 7}
all_gaussian_params = [gaussian_params_1, gaussian_params_2]
iteration_data = fake_EM(all_gaussian_params, 10)
for i in range(len(iteration_data)):
    print('Iteration:', i)
    for j in range(len(iteration_data[i])):
        print('\tGaussian', j)
        print('\t\tMean:', iteration_data[i][j]['mean'])
        print('\t\tVariance:', iteration_data[i][j]['variance'])