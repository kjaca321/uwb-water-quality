import serial
import time
import pandas as pd

ser = serial.Serial('COM7', 115200)
time.sleep(2)

path = 'live_data/calibration_data.csv'
with open(path, 'w') as file:
    file.write("Timestamp,RSSI,ToF,AvgDistance\n")  # CSV header
    count = 0
    try:
        while count < 150:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("DATA"):
                data = line.split(",")
                if len(data) == 4:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    avg_rssi = data[1]
                    avg_tof = data[2]
                    avg_distance = data[3]
                    file.write(f"{timestamp},{avg_rssi},{avg_tof},{avg_distance}\n")
                    print(f"Saved: {timestamp},{avg_rssi},{avg_tof},{avg_distance}")
                    count += 1

    except KeyboardInterrupt:
        print("Exiting...")
ser.close()

calib_data = pd.read_csv(path)
base_tof, base_rssi = calib_data['ToF'].mean(), calib_data['RSSI'].mean()

end_path = 'calibration.txt'
with open(end_path, 'w') as file:
    file.write(f"{base_tof}\n")
    file.write(f"{base_rssi}\n")