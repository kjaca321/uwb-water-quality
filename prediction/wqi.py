def calc_wqi(df):
    def normalize(value, value_range, invert=False):
        min_value, max_value = value_range
        normalized_value = ((value - min_value) / (max_value - min_value)) * 100
        return 100 - normalized_value if invert else normalized_value
    cols = ['salinity (ppt)', 'total dissolved solids (g/L)', 'temperature (deg C)']
    weights = {cols[0]: 0.4, cols[1]: 0.5, cols[2]: 0.1}
    data = df.copy()
    data['WQI'] = 0
    for col in cols:
        data['norm: ' + col] = normalize(data[col], (data[col].min(), data[col].max()), 
                                         invert=col in ['salinity (ppt)', 'total dissolved solids (g/L)'])
        data['subindex: ' + col] = data['norm: ' + col] * weights[col]
        data['WQI'] += data['subindex: ' + col]
    return data