# PINN
<<<<<<< HEAD
## Task

Step 1 Calibration, finished, output: Data/new_file.csv

Step 2 LSTM predict trajectory

Step 3 LSTM predict residual

Step 4 NN predict trajectory

Step 5 NN predict residual


## new_file.csv attributes

Time	

Speed1  :Speed of vehicle 1

E1  :East (x) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)

N1	:North (y) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)

Speed2

E2	

N2	

IVS1	:distance betweem vehicle 1 and vehicle 2 (m)

A1	:acceleration of vehicle 1

A2	

A_hat	:Predicted acceleration using physical method

V_hat	:Predicted speed using physical method

=======
Task：
Step 1 Calibration, finished, output: Data/new_file.csv
Step 2 LSTM predict trajectory
Step 3 LSTM predict residual
Step 4 NN predict trajectory
Step 5 NN predict residual


new_file.csv attributes
Time	
Speed1  :Speed of vehicle 1
E1  :East (x) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)
N1	:North (y) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)
Speed2
E2	
N2	
IVS1	:distance betweem vehicle 1 and vehicle 2 (m)
A1	:acceleration of vehicle 1
A2	
A_hat	:Predicted acceleration using physical method
V_hat	:Predicted speed using physical method
>>>>>>> 910655f38717293d758230577ff6e86a9f4b2bf8
A_error:prediction error of physical method, A_error = A_hat-A2
