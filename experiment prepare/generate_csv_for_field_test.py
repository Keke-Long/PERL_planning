import pandas as pd
import os

def process_csv_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Check if the 'v5_mpc_base2' column exists
            if 'v5_mpc_base2' in df.columns:
                # Get the 'v5_mpc_base2' column and convert it to a list
                #data_list = df['v5_mpc_base2'].dropna().tolist()
                data_list = df['v5_mpc_base2'].iloc[:230].dropna().tolist()

                # Only process if the list contains data
                if data_list:
                    # Get the first value, repeat it 20 times, and add to the start of the list
                    first_value = data_list[0]
                    # Add to the start of the list
                    prefixed_list = [first_value] * 20 + data_list

                    # Get the last value, repeat it 10 times, and add to the end of the list
                    last_value = data_list[-1]
                    print('last_value', last_value)
                    # Add to the end of the list
                    final_list = prefixed_list + [last_value] * 10
                else:
                    # If data_list is empty, create an empty list
                    final_list = []

                # Use the first number in the original filename to name the new file
                first_number = filename.split('_')[0]
                new_filename = f"{first_number}.csv"
                output_path = os.path.join(output_folder, new_filename)

                # Save the modified list as a CSV file
                pd.DataFrame(final_list).to_csv(output_path, index=False, header=False)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"'v5_mpc_base2' column not found in {filename}.")
        else:
            pass

# Set the input and output folder paths
input_folder = './data/NGSIM_I80_results/'
output_folder = './data/NGSIM_I80_field_test/'

# Call the function to process files
process_csv_files(input_folder, output_folder)
