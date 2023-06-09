import numpy as np
import matplotlib.pyplot as plt

# 加载MLP和LSTM的convergence rate数据
LSTM_convergence_rate = np.loadtxt("../LSTM/convergence_rate.csv", delimiter=",")
IDM_MLP_convergence_rate = np.loadtxt("../IDM_NN/convergence_rate.csv", delimiter=",")
IDM_LSTM_convergence_rate = np.loadtxt("../IDM_LSTM/convergence_rate.csv", delimiter=",")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(LSTM_convergence_rate, label="LSTM")
ax.plot(IDM_MLP_convergence_rate, label="PINN(IDM+MLP)")
ax.plot(IDM_LSTM_convergence_rate, label="PINN(IDM+LSTM)")
ax.set_ylabel("MSE Loss (m^2/s^4)")
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)
plt.xlabel("Epoch")
plt.ylim([0, 0.002])
plt.title("Convergence Rate Comparison")
plt.legend()
plt.savefig('Convergence Rate Comparison.png')
plt.show()
