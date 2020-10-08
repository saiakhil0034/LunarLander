# ECE 276A          : Project IV
# Students          : A53279786,A53284020
# Professor         : Prof. Nikolay Atanasov
# Last change date  : June 12th, 2020

Installation required:
Tensorflow  : pip install --user tensorflow
Keras       : pip install --user keras


runnable_files :
sarsa.py
    - for computing optimal q-value function and accordingly policy to stabilise the lunar lander using SARSA and Q-Learning algortihms. For SARSA only one line needs to be changed in update model method of SARSA class. Submission has Q-learning algorithm


# How to run the files (make sure you have data and config files in same path and make necessary changes under class defn).
python3 -u sarsa.py

# required data files
None

# for different parts of the question we just need to run the file with different configs
# Uncomment respective calling of the functions and comment unnecessary ones

# File Structure
- models
    - model_name
        -iteration
            -model files
- stats
    - model_name
        -iteration
            -stats files
- gifs
    - model_name
        -iteration
            -gifs

Note that above directories get created once you run the sarsa.py algorithsm
- utils
    - __init__.py
    - nn_model.py
    - scikit_model.py
    - ml_model.py
    - AbsoluteSolver.py
    - plotting.py
- sarsa.py
- lspi_td.py
- main.py
- PolynomialApproximator.py
- env.py
- README.md



# Files description

main.py
    - Class InvertedPendulum:
        This class has all helper functions (static methods)
        Methods available :
            __init__                : constructor
            l_xu                    : stage cost functions
            f_xu                    : motion model
            get_prob                : method to get probability for a state
            get_prob_tr             : method to get probability transition matrix
            get_probs               : get prob matrices for all control inputs
            precision_arr           : method for changin all elements to 2 decimal precision
            ticks_labels            : method to get ticks labels in plotting
            visualise               : mthod for visualising the optimal value and policy functions
            value_iteration         : method to find optimal value and policy function thorugh value iteration
            policy_iteration        : method to find optimal value and policy function thorugh policy iteration
            get_trajectory          : method for getting trajectory given a initial state from optimal policy
            plot_trajectory_ut      : for plotting the trajectory
            visualize_v             : for visualizing the value function over iterations
            get_convergence_plot    : for showing the convergence
            get_states_plots        : For plotting value of states across time

    - Class Utils:
        """class for normal Utils across code"""
        Methods available :
            __init__                : constructor
            get_z_score             : z score for 1d
            get_gaussian_p          : gaussian pdf for 1d
            get_mv_pdf              : method returning pdf method
            get_mv_gaussian         : get multivariate probability

    - Class EnvAnimate:
        Inverted Pendulum Animation Settings
        Methods available :
            __init__                : constructor
            load_random_test_trajectory : Random trajectory for example
            load_trajectory         : Once a trajectory is loaded, you can run start() to see the animation
            start                   : Set up plot for to call animate() function periodically, to animate
            map_correlation         : to find correlation between current map and current prediction


# configs
Change the configs and run for different values of them


# results
all plots are added under plots directory in submission and gifs are under gifs directory. file names corresponds to the configuration


