import os
import pandas as pd

df = pd.read_csv('/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/EMD_Reconstructed(i80 0400-0415).csv')

# Unit conversion: ft -> m, ft/s -> m/s, ft/s^2 -> m/s^2
df['Local_X'] *= 0.3048
df['Local_Y'] *= 0.3048
df['v_Vel'] *= 0.3048
df['v_Acc'] *= 0.3048
df['Space_Headway'] *= 0.3048
df['Global_Time'] *= 0.001

Duration = 250

# Initialize output data list and set of processed starting vehicles
output_data = []
processed_start_vehicles = set()

# Sort by Frame_ID column
df_sorted = df.sort_values(by='Frame_ID')

# Iterate through each row of data
for index, row in df_sorted.iterrows():

    # print('index', index)

    # Skip if the vehicle ID has already been processed as a starting vehicle
    current_vehicle_id = row['Vehicle_ID']
    if current_vehicle_id in processed_start_vehicles:
        continue

    # Store information of 6 consecutive vehicles
    chain = []
    current_vehicle_id = row['Vehicle_ID']
    valid_chain = True
    t_frame = row['Frame_ID']
    vehicle_ids = []  # Used to store the IDs of consecutive vehicles

    for n in range(6):
        #print(f'the {n} th vehicle')
        if current_vehicle_id == 0:
            break

        # Filter data for the current vehicle within the next 200 frames
        vehicle_data = df_sorted[(df_sorted['Vehicle_ID'] == current_vehicle_id) &
                                 (df_sorted['Frame_ID'] >= t_frame) &
                                 (df_sorted['Frame_ID'] <= t_frame + Duration)]

        # Check if there is continuous data
        if vehicle_data.empty or len(vehicle_data) < Duration:
            valid_chain = False
            #print('No continuous data')
            break

        # Check if the 'Following' column is consistent
        if not all(vehicle_data['Following'] == vehicle_data.iloc[0]['Following']):
            valid_chain = False
            #print('Following is inconsistent')
            break

        # Record data
        for _, data_row in vehicle_data.iterrows():
            chain.append({
                't': data_row['Global_Time'],
                'veh_ID': current_vehicle_id,
                'Y': data_row['Local_Y'],
                'v': data_row['v_Vel'],
                'a': data_row['v_Acc']
            })

        # Add the current vehicle ID to the list for the filename
        vehicle_ids.append(str(int(current_vehicle_id)))

        # Move to the next vehicle
        current_vehicle_id = vehicle_data.iloc[0]['Following']

    # If successfully found 6 vehicles
    if len(vehicle_ids) == 6 and valid_chain:
        # Add the starting vehicle ID to the processed set
        processed_start_vehicles.add(int(vehicle_ids[0]))

        # Initialize the output data structure
        output_data = {}
        output_data['t1'] = []  # Keep only one time column
        for i in range(1, 7):  # Create 6 sets of data columns
            output_data[f'ID{i}'] = []
            output_data[f'Y{i}'] = []
            output_data[f'v{i}'] = []
            output_data[f'a{i}'] = []

        # Get the minimum position of the 6th vehicle
        last_vehicle_data = df_sorted[(df_sorted['Vehicle_ID'] == int(vehicle_ids[-1])) &
                                      (df_sorted['Frame_ID'] >= t_frame) &
                                      (df_sorted['Frame_ID'] <= t_frame + Duration)]
        min_y_6 = last_vehicle_data['Local_Y'].min()

        # Iterate through each ID and collect data
        for i, vid in enumerate(vehicle_ids):
            vehicle_data = df_sorted[(df_sorted['Vehicle_ID'] == int(vid)) &
                                     (df_sorted['Frame_ID'] >= t_frame) &
                                     (df_sorted['Frame_ID'] <= t_frame + Duration)]

            min_time = vehicle_data['Global_Time'].min()  # Find the minimum time
            for data_row in vehicle_data.itertuples():
                if i == 0:  # Add time only for the first set of data
                    relative_time = data_row.Global_Time - min_time
                    output_data['t1'].append(round(relative_time, 4))
                output_data[f'ID{i + 1}'].append(int(vid))
                output_data[f'Y{i + 1}'].append(round(data_row.Local_Y - min_y_6, 4))
                output_data[f'v{i + 1}'].append(round(data_row.v_Vel, 4))
                output_data[f'a{i + 1}'].append(round(data_row.v_Acc, 4))

        # Format and save the data
        output_df = pd.DataFrame(output_data)
        file_name = f"./data/NGSIM_I80/{'_'.join(vehicle_ids)}.csv"
        output_df.to_csv(file_name, index=False)
        print(f"Data for vehicles {', '.join(vehicle_ids)} saved to {file_name}.")
