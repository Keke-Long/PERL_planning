import numpy as np
import pandas as pd

def IDM(arg, vi, delta_v, delta_d):
    vf, A, b, s0, T = arg
    s_star = s0 + np.max([0, vi * T + (vi * delta_v) / (2 * (A * b) ** 0.5)], axis=0)
    epsilon = 1e-5
    ahat = A * (1 - (vi / vf) ** 4 - (s_star / (delta_d + epsilon)) ** 2)
    return ahat

# Load data
file_name = '1_11_24_44_55_54.csv'
df = pd.read_csv(f'./data/NGSIM_I80_results/{file_name}')

# Set IDM model parameters, these may need to be adjusted based on actual conditions
arg_base = (20, 1.5, 2, 1.3, 1.5)  # Example parameters: vf, A, b, s0, T

# Initialize the trajectory columns for the following vehicle
df['a6_mpc_base2'] = 0
df['v6_mpc_base2'] = df.loc[0, 'v6']  # Assume the initial speed is the same as the first row
df['Y6_mpc_base2'] = df.loc[0, 'Y6']  # Assume the initial position is the same as the first row

df['a6_mpc_base3'] = 0
df['v6_mpc_base3'] = df.loc[0, 'v6']
df['Y6_mpc_base3'] = df.loc[0, 'Y6']

# Iterate through time steps to update states
dt = 0.1  # Time step
for i in range(1, len(df)):
    df.loc[i, 'a6_mpc_base2'] = IDM(arg_base, df.loc[i-1, 'v6_mpc_base2'], df.loc[i-1, 'v5_mpc_base2'] - df.loc[i-1, 'v6_mpc_base2'], df.loc[i-1, 'Y5_mpc_base2'] - df.loc[i-1, 'Y6_mpc_base2'])
    df.loc[i, 'v6_mpc_base2'] = df.loc[i-1, 'v6_mpc_base2'] + df.loc[i, 'a6_mpc_base2'] * dt
    df.loc[i, 'Y6_mpc_base2'] = df.loc[i-1, 'Y6_mpc_base2'] + df.loc[i-1, 'v6_mpc_base2'] * dt

    df.loc[i, 'a6_mpc_base3'] = IDM(arg_base, df.loc[i-1, 'v6_mpc_base3'], df.loc[i-1, 'v5_mpc_base3'] - df.loc[i-1, 'v6_mpc_base3'], df.loc[i-1, 'Y5_mpc_base3'] - df.loc[i-1, 'Y6_mpc_base3'])
    df.loc[i, 'v6_mpc_base3'] = df.loc[i-1, 'v6_mpc_base3'] + df.loc[i, 'a6_mpc_base3'] * dt
    df.loc[i, 'Y6_mpc_base3'] = df.loc[i-1, 'Y6_mpc_base3'] + df.loc[i-1, 'v6_mpc_base3'] * dt

# Save the updated CSV
df.to_csv(f'./data/NGSIM_I80_results/{file_name}', index=False)
