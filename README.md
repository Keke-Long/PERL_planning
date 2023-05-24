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




## Results of Platoon 1 CAV
### Physical model
IDM arg = (23, 0.51, 4, 3.5, 0.56)

MSE when predicting acceleration: 0.16

![Image text](https://github.com/Keke-Long/PINN/blob/main/Physical_model/Platoon1_IDM_result_comparison.jpg)
 

### LSTM model
MSE when predicting acceleration: 0.123193

![Image text](https://github.com/Keke-Long/PINN/blob/main/LSTM/Platoon1_LSTM_result.png)


### PINN(IDM+LSTM)
MSE when predicting acceleration: 0.11098

![Image text](https://github.com/Keke-Long/PINN/blob/main/IDM%2BLSTM/Platoon1_PINN_result_plot.png)



## Results of Platoon 3 HV
### Physical model
IDM arg = (25.4,   1.3,  4.0,  3.9,  1.57)

MSE when predicting acceleration: 0.09978

![Image text](https://github.com/Keke-Long/PINN/blob/main/Physical_model/Platoon3_IDM_result_comparison.jpg)
 

### LSTM model
MSE when predicting acceleration: 0.00372

![Image text](https://github.com/Keke-Long/PINN/blob/main/LSTM/Platoon3_LSTM_result.png)


### PINN(IDM+LSTM)
MSE when predicting acceleration: 0.22685

![Image text](https://github.com/Keke-Long/PINN/blob/main/IDM%2BLSTM/Platoon3_PINN_result_plot.png)


