import pandas
from datetime import datetime
import pickle


def get_and_pickle_data(df, pickled_name, attack_name):
    attack_rows = df.loc[df["Label"] == attack_name]
    key_data = []
    flow_duration = None
    for i in range(75):
        summed_value = attack_rows.iloc[:, i + 3].astype(float).sum()
        if i == 0:
            flow_duration = summed_value
        elif i == 13 or i == 14:
            # Ignoring total packets and bytes because of infinity and NaN issues
            pass
        else:
            key_data.append(summed_value)
    list_to_pickle = [x / flow_duration for x in key_data]
    with open(pickled_name, "wb") as f:
        pickle.dump(list_to_pickle, f)
    check_pickle_success(pickled_name, list_to_pickle)
    print(list_to_pickle)


def check_pickle_success(pickled_name, variable):
    with open(pickled_name, "rb") as f:
        pickled_var = pickle.load(f)
    assert pickled_var == variable


# First Infiltration Attack
df = pandas.read_csv("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv")
attack_name = "Infilteration"
pickled_name = "infil1.pckl"
get_and_pickle_data(df, pickled_name, attack_name)

# Second Infiltration Attack
attack_name = "Infilteration"
df = pandas.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "infil2.pckl"
get_and_pickle_data(df, pickled_name, attack_name)

# First DoS Attack - GoldenEye
df = pandas.read_csv("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "dos1.pckl"
attack_name = "DoS attacks-GoldenEye"
get_and_pickle_data(df, pickled_name, attack_name)

# Second DoS Attack - Slowloris
df = pandas.read_csv("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "dos2.pckl"
attack_name = "DoS attacks-Slowloris"
get_and_pickle_data(df, pickled_name, attack_name)

# Third DoS Attack - Hulk
df = pandas.read_csv("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "dos3.pckl"
attack_name = "DoS attacks-Hulk"
get_and_pickle_data(df, pickled_name, attack_name)

# First SQL Injection Attack
df = pandas.read_csv("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "sqlinjection1.pckl"
attack_name = "SQL Injection"
get_and_pickle_data(df, pickled_name, attack_name)

# Second SQL Injection Attack
df = pandas.read_csv("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "sqlinjection2.pckl"
attack_name = "SQL Injection"
get_and_pickle_data(df, pickled_name, attack_name)

# First Benign Data Set
df = pandas.read_csv("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "benign1.pckl"
attack_name = "Benign"
get_and_pickle_data(df, pickled_name, attack_name)

# Second Benign Data Set
df = pandas.read_csv("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "benign2.pckl"
attack_name = "Benign"
get_and_pickle_data(df, pickled_name, attack_name)

# Third Benign Data Set
df = pandas.read_csv("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "benign3.pckl"
attack_name = "Benign"
get_and_pickle_data(df, pickled_name, attack_name)

# Fourth Benign Data Set
df = pandas.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "benign4.pckl"
attack_name = "Benign"
get_and_pickle_data(df, pickled_name, attack_name)

# Fifth Benign Data Set
df = pandas.read_csv("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "benign5.pckl"
attack_name = "Benign"
get_and_pickle_data(df, pickled_name, attack_name)

# Bot Data Set
df = pandas.read_csv("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "bot1.pckl"
attack_name = "Bot"
get_and_pickle_data(df, pickled_name, attack_name)

# Brute Force - Web Data Set
df = pandas.read_csv("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "bf1.pckl"
attack_name = "Brute Force -Web"
get_and_pickle_data(df, pickled_name, attack_name)

# Brute Force - XSS Data Set
df = pandas.read_csv("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv")
pickled_name = "bf2.pckl"
attack_name = "Brute Force -XSS"
get_and_pickle_data(df, pickled_name, attack_name)
