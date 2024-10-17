import numpy as np
from sklearn.linear_model import Ridge
from pykalman import KalmanFilter

def simulator(data, sim_data, noise_percent_tof=2.0, noise_percent_rssi=2.0):
    coeffs = Ridge().fit(data[['salinity (ppt)', 'total dissolved solids (g/L)']], 
                        data[['del_ToF', 'del_RSSI']]).coef_
    def sim_del_tof(df):
        return coeffs[0][0] * df['salinity (ppt)'] + coeffs[0][1] * df['total dissolved solids (g/L)']
    def sim_del_rssi(df):
        return coeffs[1][0] * df['salinity (ppt)'] + coeffs[1][1] * df['total dissolved solids (g/L)']

    sim_data = sim_data.copy()
    sim_data['del_ToF'] = sim_del_tof(sim_data)
    sim_data['del_RSSI'] = sim_del_rssi(sim_data)


    mean_tof = abs(sim_data['del_ToF'].mean())
    mean_rssi = abs(sim_data['del_RSSI'].mean())

    np.random.seed(42)  # For reproducibility
    sim_data['del_ToF'] += np.random.normal(0, (noise_percent_tof / 100) * mean_tof, sim_data.shape[0])
    sim_data['del_RSSI'] += np.random.normal(0, (noise_percent_rssi / 100) * mean_rssi, sim_data.shape[0])


    def kalman_filter(sim_data):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                        transition_matrices=[1], 
                        observation_matrices=[1], 
                        transition_covariance=.2,  # Increased process noise
                        observation_covariance=.15,
                        initial_state_covariance=1e5)
        state_means, _ = kf.smooth(sim_data)
        return state_means.flatten()

    # Apply filters
    sim_data['Kalman_ToF'] = kalman_filter(sim_data['del_ToF'])
    sim_data['Kalman_RSSI'] = kalman_filter(sim_data['del_RSSI'])

    print("Average ToF:", sim_data['del_ToF'].mean())
    print("Min/max ToF: ", sim_data['del_ToF'].min(), sim_data['del_ToF'].max())
    print("Stdev ToF:", sim_data['del_ToF'].std())
    print("Average RSSI:", sim_data['del_RSSI'].mean())
    print("Min/max RSSI: ", sim_data['del_RSSI'].min(), sim_data['del_RSSI'].max())
    print("Stdev RSSI: ", sim_data['del_RSSI'].std())

    return sim_data