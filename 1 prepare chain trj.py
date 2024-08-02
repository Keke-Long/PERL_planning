import os
import pandas as pd

df = pd.read_csv('./I-80_download/EMD_Reconstructed(i80 0400-0415).csv')

# Unit conversion: feet to meters, feet/s to m/s, feet/s^2 to m/s^2
df['Local_X'] *= 0.3048
df['Local_Y'] *= 0.3048
df['v_Vel'] *= 0.3048
df['v_Acc'] *= 0.3048
df['Space_Headway'] *= 0.3048
df['Global_Time'] *= 0.001

Duration = 250

# Initialize output data list and set of processed start vehicles
output_data = []
processed_start_vehicles = set()

# Sort the dataframe by Frame_ID
df_sorted = df.sort_values(by='Frame_ID')

# Iterate over each row of the dataframe
for index, row in df_sorted.iterrows():
    # Skip if this vehicle ID has already been processed as a start vehicle
    current_vehicle_id = row['Vehicle_ID']
    if current_vehicle_id in processed_start_vehicles:
        continue

    # Store information for a chain of 6 consecutive vehicles
    chain = []
    current_vehicle_id = row['Vehicle_ID']
    valid_chain = True
    t_frame = row['Frame_ID']
    vehicle_ids = []  # Store the vehicle IDs for the chain

    for n in range(6):
        if current_vehicle_id == 0:
            break

        # Filter data for the current vehicle within 200 frames from the start frame
        vehicle_data = df_sorted[(df_sorted['Vehicle_ID'] == current_vehicle_id) &
                                 (df_sorted['Frame_ID'] >= t_frame) &
                                 (df_sorted['Frame_ID'] <= t_frame + Duration)]

        # Check for continuous data availability
        if vehicle_data.empty or len(vehicle_data) < Duration:
            valid_chain = False
            break

        # Ensure the 'Following' field is consistent
        if not all(vehicle_data['Following'] == vehicle_data.iloc[0]['Following']):
            valid_chain = False
            break

        # Store the data points for the current vehicle
        for _, data_row in vehicle_data.iterrows():
            chain.append({
                't': data_row['Global_Time'],
                'veh_ID': current_vehicle_id,
                'Y': data_row['Local_Y'],
                'v': data_row['v_Vel'],
                'a': data_row['v_Acc']
            })

        # Append the current vehicle ID to the list for use in the filename
        vehicle_ids.append(str(int(current_vehicle_id)))

        # Move to the next vehicle in the chain
        current_vehicle_id = vehicle_data.iloc[0]['Following']

    # If a valid chain of 6 vehicles was formed
    if len(vehicle_ids) == 6 and valid_chain:
        processed_start_vehicles.add(int(vehicle_ids[0]))

        # Initialize the output data structure
        output_data = {}
        output_data['t1'] = []  # Store only one time column
        for i in range(1, 7):  # Create columns for 6 vehicles
            output_data[f'ID{i}'] = []
            output_data[f'Y{i}'] = []
            output_data[f'v{i}'] = []
            output_data[f'a{i}'] = []

        # Process and save the data
        output_df = pd.DataFrame(output_data)
        file_name = f"./data/NGSIM_I80/{'_'.join(vehicle_ids)}.csv"
        output_df.to_csv(file_name, index=False)
        print(f"Data for vehicles {', '.join(vehicle_ids)} saved to {file_name}.")
