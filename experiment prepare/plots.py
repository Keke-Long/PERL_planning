import os
import pandas as pd
import matplotlib.pyplot as plt

# Set folder path
folder_path = './NGSIM_I80_field_test_base2'  # Please replace this path with your folder path

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop through each CSV file and process the data
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)

    # Read the CSV file
    try:
        df = pd.read_csv(file_path, header=None, names=['Speed'])
        print(f"File {file_name} read successfully!")
    except Exception as e:
        print(f"Error reading the file {file_name}: {e}")
        continue

    # Display the first few rows of the data
    print(df.head())

    # Calculate the time column
    df['Time'] = df.index * 0.1

    # Calculate acceleration
    df['Acceleration'] = df['Speed'].diff() / 0.1

    # Calculate position
    df['Position'] = df['Speed'].cumsum() * 0.1

    # Plot acceleration, speed, and position
    plt.figure(figsize=(12, 8))

    # Acceleration plot
    plt.subplot(3, 1, 1)
    plt.plot(df['Time'], df['Acceleration'], label='Acceleration', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration vs Time')
    plt.legend()

    # Speed plot
    plt.subplot(3, 1, 2)
    plt.plot(df['Time'], df['Speed'], label='Speed', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed vs Time')
    plt.legend()

    # Position plot
    plt.subplot(3, 1, 3)
    plt.plot(df['Time'], df['Position'], label='Position', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    output_file_path = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}.png")
    plt.savefig(output_file_path)
    plt.close()
    print(f"Plot saved to {output_file_path}")
