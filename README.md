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

![Image text](https://github.com/Keke-Long/PINN/blob/main/Physical_model/Platoon1_IDM_result.jpg)
 

### LSTM model

![Image text](https://github.com/Keke-Long/PINN/blob/main/LSTM/Platoon1_LSTM_result.png)


### PINN(IDM+LSTM)

![Image text](https://github.com/Keke-Long/PINN/blob/main/IDM%2BLSTM/Platoon1_PINN_result.png)



## Results of Platoon 3 HV
### Physical model

![Image text](https://github.com/Keke-Long/PINN/blob/main/Physical_model/Platoon3_IDM_result.jpg)
 

### LSTM model

![Image text](https://github.com/Keke-Long/PINN/blob/main/LSTM/Platoon3_LSTM_result.png)


### PINN(IDM+LSTM)

![Image text](https://github.com/Keke-Long/PINN/blob/main/IDM%2BLSTM/Platoon3_PINN_result.png)


