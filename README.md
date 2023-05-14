# PINN

## Task

Step 1 Calibration, finished, output: Data/new_file.csv

Step 2 LSTM predict trajectory, finished,

Step 3 LSTM predict residual, finished,

Step 4 NN predict trajectory

Step 5 NN predict residual


## new_file.csv attributes

Time	

Speed1  :Speed of vehicle 1 (m/s)

E1  :East (x) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)

N1	:North (y) coordinate of vehicle 1 in the local ENU plane (common center for all vehicles) (m)

Speed2

E2	

N2	

IVS1	:distance between vehicle 1 and vehicle 2 (m)

A1	:acceleration of vehicle 1 (m/s^2)

A2	:acceleration of vehicle 2 (m/s^2)

A_hat	:Predicted acceleration using physical method (m/s^2)

V_hat	:Predicted speed using physical method (m/s)


## Results
### Physical model
IDM arg = (21.25, 0.53, 1.53, 5, 0.72)
MSE when predicting acceleration: 0.23622273396526933 
![Image text](https://github.com/Keke-Long/PINN/blob/main/Physical_model/IDM_result_comparison.jpg)
 

### LSTM model
MSE when predicting acceleration: 0.00028513008160522635 
![Image text](https://github.com/Keke-Long/PINN/blob/main/LSTM/LSTM_result_plot.png)


### PINN(IDM+LSTM)
MSE when PINN predicting error: 0.00024908116675127724
train_size = 12585
MSE when predicting acceleration: 0.2920514412228489 
![Image text](https://github.com/Keke-Long/PINN/blob/main/IDM%2BLSTM/PINN_result_plot.png)