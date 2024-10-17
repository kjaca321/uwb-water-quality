import serial
import time
import pandas as pd
from pykalman import KalmanFilter
import joblib

calib_path = 'calibration.txt'
with open(calib_path, 'r') as file:
    lines = file.readlines()
base_tof = float(lines[0].strip())
base_rssi = float(lines[1].strip())

ser = serial.Serial('COM7', 115200)
time.sleep(2)

path = '../trials/static_live.csv'
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

live_data = pd.read_csv(path)
live_data['del_ToF'] = live_data['ToF'] - base_tof
live_data['del_RSSI'] = live_data['RSSI'] - base_rssi

del_tof_mean = live_data['del_ToF'].mean()
del_rssi_mean = live_data['del_RSSI'].mean()

print('del tof mean: ', del_tof_mean)
print('del rssi mean: ', del_rssi_mean)

def kalman_filter(live_data):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                      transition_matrices=[1], 
                      observation_matrices=[1], 
                      transition_covariance=1e-2,  # Increased process noise
                      observation_covariance=1e-1)
    state_means, _ = kf.smooth(live_data)
    return state_means.flatten()

# Apply filters
live_data['Kalman_ToF'] = kalman_filter(live_data['del_ToF'])
live_data['Kalman_RSSI'] = kalman_filter(live_data['del_RSSI'])


model = joblib.load('../prediction/saved_models/rfr_model.pxl')
scaler = joblib.load('../prediction/saved_models/scaler.pxl')
poly = joblib.load('../prediction/saved_models/poly_features.pxl')

test_data = live_data[['del_ToF', 'Kalman_ToF', 'del_RSSI', 'Kalman_RSSI']]
# test_data = live_data[['del_ToF', 'del_RSSI']]

test_data_scaled = scaler.transform(test_data)
test_data_poly = poly.transform(test_data_scaled)
predictions = model.predict(test_data_poly)
output = pd.DataFrame(predictions, 
                           columns=['salinity (ppt)', 'total dissolved solids (g/L)', 'del temp (deg C)'])

mean_salinity = output['salinity (ppt)'].mean()
mean_tds = output['total dissolved solids (g/L)'].mean()
output['temperature (deg C)'] = output['del temp (deg C)'] + 24
mean_temp = output['temperature (deg C)'].mean()

salinity = round(max(mean_salinity, 0.00), 4)
tds = salinity + round(max(mean_tds, 0.00), 4)
temp = round(mean_temp, 4)

print('salinity: ', salinity, ' ppt')
print('TDS: ', tds, ' g/L')
print('temperature: ', temp, ' deg C')

result_data = pd.read_csv('../trials/result_data.csv')
result_id = ''
new_result = pd.DataFrame({'ID': [result_id], 'salinity (ppt)': [salinity], 
                           'total dissolved solids (g/L)': [tds], 'temperature (deg C)': [temp]})
result_data = pd.concat([result_data, new_result], ignore_index=True)
result_data.to_csv('../trials/result_data.csv', index=False)