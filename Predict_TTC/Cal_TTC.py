import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/ASta_050719_platoon1.csv', skiprows=range(5))

df['TTC'] = df['Alt1'] / (df['Speed1'] - df['Speed2'])

plt.scatter(df['Time'], df['TTC'], label='TTC')
plt.legend()
plt.xlabel('Time')
plt.ylabel('TTC')
plt.ylim(-50, 50)
plt.title('Time to Collision (TTC) vs Time')
plt.savefig('TTC for platoon 1.jpg')
plt.show()
