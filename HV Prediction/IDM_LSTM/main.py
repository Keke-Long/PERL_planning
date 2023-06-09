import subprocess
import argparse

platoon_num = 1

# Run train.py
train_process = subprocess.Popen(['python', 'train.py', '--platoon_num', str(platoon_num)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("train.py output:")
while True:
    output = train_process.stdout.readline()
    if output == b'' and train_process.poll() is not None:
        break
    if output:
        print(output.decode().strip())

train_output, train_error = train_process.communicate()
print("train.py error:")
print(train_error.decode())

# Run predict.py
predict_process = subprocess.Popen(['python', 'predict.py', '--platoon_num', str(platoon_num)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("predict.py output:")
while True:
    output = predict_process.stdout.readline()
    if output == b'' and predict_process.poll() is not None:
        break
    if output:
        print(output.decode().strip())

predict_output, predict_error = predict_process.communicate()
print("predict.py error:")
print(predict_error.decode())
