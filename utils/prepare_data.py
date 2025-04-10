import mne
import pandas as pd

def make_raw(csv_path, duration):

    # Carregar o arquivo CSV
    df = pd.read_csv(csv_path)
    df_timestamp_events = df[['timestamp', 'events']]
    df_timestamp_events = df_timestamp_events.loc[df_timestamp_events['events'] != 0]

    print(df_timestamp_events)

    # Criar dataframe com as outras colunas
    df_data = df.drop(columns=['timestamp', 'events'])

    # Convert pandas dataframe to numpy array
    data_array = df_data.to_numpy()

    ch_names = df_data.columns.tolist()

    # Create mne RawArray object
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data_array.T, info)

    for event in df_timestamp_events.values:
        if event[1] == 1:
            raw.annotations.append(event[0], 5, 'move')
        else:
            raw.annotations.append(event[0], 15, 'rest')

    return raw