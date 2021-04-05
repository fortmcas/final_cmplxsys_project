import pandas
from datetime import datetime
import pickle

def get_and_pickle_data(attack_start, attack_end, df, pickled_name):
    df['Timestamp'] = pandas.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='ignore')
    infiltration_rows_indices = df.loc[df['Label'] == 'Infilteration'].index
    selected_timestamps = []
    for timestamp in df.loc[infiltration_rows_indices]['Timestamp']:
        datetime_obj = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S')
        if ((datetime_obj >= attack_start) and (datetime_obj <= attack_end)):
            selected_timestamps += [timestamp]
    final_rows = df.loc[(df['Timestamp'].isin(selected_timestamps)) & (df['Label'] == 'Infilteration')]
    key_data = []
    flow_duration = None
    for i in range(75):
        summed_value = final_rows.iloc[:,i+3].astype(float).sum()
        if i==0:
            flow_duration=summed_value
        elif (i==13 or i==14):
            # Ignoring total packets and bytes because of infinity and NaN issues
            pass
        else:
            key_data.append(summed_value)
    list_to_pickle = [x / flow_duration for x in key_data]
    with open(pickled_name, 'wb') as f:
        pickle.dump(list_to_pickle, f)
    check_pickle_success(pickled_name, list_to_pickle)
    print(list_to_pickle)

def check_pickle_success(pickled_name, variable):
    with open(pickled_name, 'rb') as f:
        pickled_var = pickle.load(f)
    assert pickled_var == variable

# First Infiltration Attack
attack_start = datetime(2018, 2, 28, 10, 50)
attack_end = datetime(2018, 2, 28, 12, 5)
df = pandas.read_csv("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "firstinfil.pckl"
get_and_pickle_data(attack_start, attack_end, df, pickled_name)

# Second Infiltration Attack
attack_start = datetime(2018, 3, 1, 9, 57)
attack_end = datetime(2018, 3, 1, 10, 55)
df = pandas.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "secondinfil.pckl"
get_and_pickle_data(attack_start, attack_end, df, pickled_name)




