###########
#TO DO LIST
#   1.0 Possibilitar opção de salvar como EDF
#   2
#   (OK) 1.1 Refatorar os Timestamps (UNIX EPOCH não parece legal, usar o tempo relativo (0.00, 0.01 etc.))
#2 Visualizar e Editar infos de Usuário
#3 Iniciar o programa com o último usuário logado
#4 Adicionar automáticamente as boards do JSON no código
#999 REFATORAR ESTA MERDA
###########

import FreeSimpleGUI as sg

from modals.knn_modal import *
from modals.mlp_modal import *
from modals.svm_modal import *
from modals.lda_modal import *
from modals.csp_modal import *
from modals.graph_modal import *
from modals.pipeline_modal import *
from modals.user_modal import *

from utils.prepare_data import make_raw
from utils.beep import beep
import json, os, threading
import serial.tools.list_ports
from time import sleep
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

tema_medico = {
    'BACKGROUND': '#ccffcc',  # Cor de fundo
    'TEXT': '#000000',        # Cor do texto
    'INPUT': '#FFFFFF',       # Cor de fundo dos campos de entrada
    'TEXT_INPUT': '#000000',  # Cor do texto nos campos de entrada
    'SCROLL': '#FFFFFF',      # Cor de fundo da barra de rolagem
    'BUTTON': ('#0F0F0F', '#1aff66'),  # Cor de fundo e texto dos botões
    'PROGRESS': ('#000000', '#FFFFFF'),  # Cor da barra de progresso
    'BORDER': 0.5,              # Largura da borda
    'SLIDER_DEPTH': 0,        # Profundidade do controle deslizante
    'PROGRESS_DEPTH': 0,      # Profundidade da barra de progresso
}

# Defina o tema
sg.theme_add_new('TemaMedico', tema_medico)

# Use o tema personalizado
sg.theme('TemaMedico')

# Abre Json com imagens em formato de string
with open(os.path.join('config', 'icons.json'), 'r') as f:
    icons = json.load(f)

with open(os.path.join('config', 'boards.json'), 'r') as f:
    boards = json.load(f)

users_folder = os.path.join('users','data')
menu_def = [['Perfil de Coleta', ['Novo Perfil', 'Mudar Perfil']]]

colu_esq = [[sg.Text('Classificação', justification='center', font=('Helvetica', 18))],
            [sg.Input(key="-file-", change_submits=True, readonly=True),
            sg.FileBrowse(initial_folder=users_folder, file_types=(("Dados de EEG", "*.csv *.fif *.edf"),("CSV", "*.csv"),("FIF", "*.fif"),("EDF", "*.edf")), key="-file-")],
            [sg.Text("Extração de Características")],
            [sg.Combo(["CSP"], key='-feature_extraction-', readonly=True, size=(20,1)),
            sg.Button("Adicionar", key='-add_feat-')],
            [sg.Text("Classificador")],
            [sg.Combo(["LDA", "SVM", "MLP", "KNN"], key='-classification-', readonly=True, size=(20,1)),
            sg.Button("Adicionar", key='-add_class-')],
            [sg.Multiline(size=(45, 10), key='-console-')],
            [sg.Button("Ver pipeline", key='-ver-'),
            sg.Button("Executar", key='-run-'),
            sg.Button("Ver gráficos", key='-graph-')]]

colu_dir = [[sg.Text('Coleta', justification='center', font=('Helvetica', 18))],
            [sg.Text('Porta')],
            [sg.Combo([], key='-ports-', size=(30,1), enable_events=True, readonly=True), sg.Button('', key='-refresh_ports-', image_data=icons['refresh'], image_size=(25,25), image_subsample=18)],
            [sg.Text('Placa')], [sg.Combo(['Cyton'], key='-board-', size=(30,1), enable_events=True, readonly=True)],
            [sg.Button('Iniciar Coleta', key='-start-'),
            sg.Button('', key='-sound-', image_data=icons['soundOn'], image_size=(25,25), image_subsample=18),
            sg.Radio('', 'sound', default=True, key='-soundOn-', visible=False)],
            [sg.Image(data=icons['idle'], key='-visual_guide-', subsample=4)]]

layout = [  [sg.Menu(menu_def)],
            [sg.Text('Perfil de Coleta ativo:'),
             sg.Text('--', key='-logged_user-', size=(50,1)),
             sg.Text('Última sessão:'),
             sg.Text('__/__/__', key='-last_visit-')],
            [sg.HSeparator()],
            [sg.Column(colu_esq), sg.VSeparator(), sg.vtop(sg.Column(colu_dir))]]

pipeline = list()
pipeline_args = list()
logged_user = None

def executar(window, values, pipeline):

    import numpy as np
    from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import confusion_matrix

    from mne import Epochs, events_from_annotations, pick_types
    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, read_raw_fif

    tmin, tmax = -1.0, 4.0 
    # ESSA INFO DEVE VIR DO PERFIL
    event_id = dict(T1=0, T2=1)

    if '.csv' in values['-file-']:
        raw = make_raw(values['-file-'], 5)
    elif '.edf' in values['-file-']:
        raw = read_raw_edf(values['-file-'], preload=True)
    elif '.fif' in values['-file-']:
        raw = read_raw_fif(values['-file-'], preload=True)

    eegbci.standardize(raw)
    # ESSA INFO DEVE VIR DO PERFIL
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # ISSO DEVE SER UM PARÂMETRO
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, _ = events_from_annotations(raw, event_id=dict(rest=0, move=1))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1] - 2

    scores = []
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2)

    clf = Pipeline(pipeline)

    # A SCORE UTILIZADA DEVE SER UM PARÂMETRO   
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)
    predicts = cross_val_predict(clf, epochs_data_train, labels, n_jobs=None)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    window['-console-'].print("Acurácia da Classificação: %f / Distribuição de Classes: %f" % (np.mean(scores), class_balance))
    return confusion_matrix(labels, predicts)

def coletar(window, values, user):

    with open(os.path.join('config', 'icons.json'), 'r') as f:
        icons = json.load(f)

    with open(os.path.join('config', 'boards.json'), 'r') as f:
        boards = json.load(f)
    
    params = BrainFlowInputParams()
    params.serial_port = values['-ports-']

    board_id = boards[values['-board-']]

    # Crie uma instância do BoardShim
    board = BoardShim(board_id, params)

    # Prepare o quadro para a coleta de dados
    board.prepare_session()

    sleep(1)

    # Comece a transmitir dados
    board.start_stream()

    for run in range(user['numero_runs']):

        board.insert_marker(1)

        #repouso
        if values['-soundOn-']:
            beep(1000, 50)
        window['-visual_guide-'].Update(data=icons['stillPerson'], subsample=4)
        sleep(user['tempo_descanso']-0.05)

        board.insert_marker(2)

        #ação
        if values['-soundOn-']:
            beep(2000, 50)
        window['-visual_guide-'].Update(data=icons['movingPerson'], subsample=4)
        sleep(user['tempo_imagetica']-0.05)
        
    window['-visual_guide-'].Update(data=icons['idle'], size=(100,100), subsample=4)

    data = board.get_board_data()

    # Encerre a transmissão de dados
    board.stop_stream()
    board.release_session()

    eeg_data = [data[i] for i in range(1,user['number_channels']+1)]
    timestamps = data[len(data)-2]
    events = data[len(data)-1]

    #normalização dos dados de tempo
    timestamps = [i - timestamps[0] for i in timestamps]

    user_data_file = path.join('users', f'{user["nome"]}.csv')

    with open(user_data_file, mode='a', newline='') as arquivo_csv:
        writer = csv.writer(arquivo_csv)

        # Iterar sobre as listas e escrever os dados no arquivo CSV
        for i in range(len(timestamps)):
            linha = [timestamps[i]] + [eeg_data_channel[i] for eeg_data_channel in eeg_data] + [events[i]]
            writer.writerow(linha)

db = TinyDB(path.join('users', 'users.json'))
User = Query()

def main():

    window = sg.Window('Projeto TCC', layout, finalize=True)
    conf_mat = [[0,0],[0,0]]

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == '-add_feat-':
            if values['-feature_extraction-'] == "CSP":
                method = CSP_modal()
                response = method.open_window()
                if response != ('cancel'):
                    pipeline.append(response['method'])
                    pipeline_args.append(response['args'])
                    window['-console-'].print(f"CSP adicionado com os seguintes parâmetros: {response['args']}")
        if event == '-add_class-':
            if values['-classification-'] == "LDA":
                method = LDA_modal()
                response = method.open_window()
                if response != ('cancel'):
                    pipeline.append(response['method'])
                    pipeline_args.append(response['args'])
                    window['-console-'].print(f"LDA adicionado com os seguintes parâmetros: {response['args']}")
            elif values['-classification-'] == "SVM":
                method = SVM_modal()
                response = method.open_window()
                if response != ('cancel'):
                    pipeline.append(response['method'])
                    pipeline_args.append(response['args'])
                    window['-console-'].print(f"SVM adicionado com os seguintes parâmetros: {response['args']}")
            elif values['-classification-'] == "MLP":
                method = MLP_modal()
                response = method.open_window()
                if response != ('cancel'):
                    pipeline.append(response['method'])
                    pipeline_args.append(response['args'])
                    window['-console-'].print(f"MLP adicionado com os seguintes parâmetros: {response['args']}")
            elif values['-classification-'] == "KNN":
                method = KNN_modal()
                response = method.open_window()
                if response != ('cancel'):
                    pipeline.append(response['method'])
                    pipeline_args.append(response['args'])
                    window['-console-'].print(f"KNN adicionado com os seguintes parâmetros: {response['args']}")
        if event == '-ver-':
            ppl = Pipeline_modal(pipeline, pipeline_args)
            updates = ppl.open_window()
            for pos in updates:
                removed = pipeline.pop(pos)
                pipeline_args.pop(pos)
                window['-console-'].print(f'{removed[0]} removido do Pipeline')
        if event == '-run-':
            try:
                conf_mat = executar(window, values, pipeline)
            except Exception as e:
                window['-console-'].print("Erro:", e)
        if event == 'Novo Perfil':
            modal = NewUser()
            user = modal.open_window()
            if user:
                logged_user = user[0]
                window['-logged_user-'].Update(user[0])
                window['-last_visit-'].Update(user[1])
        if event == 'Mudar Perfil':
            modal = ChangeUser()
            user = modal.open_window()
            if user:
                logged_user = user[0]
                window['-logged_user-'].Update(user[0])
                window['-last_visit-'].Update(user[1])
        if event == '-refresh_ports-':
            ports = [port.name for port in serial.tools.list_ports.comports()]
            window['-ports-'].Update(values=ports)
        if event == '-sound-':
            if values['-soundOn-']:
                window['-soundOn-'].Update(value=False)
                window['-sound-'].Update(image_data=icons['soundOff'], image_size=(25,25), image_subsample=18)
            else:
                window['-soundOn-'].Update(value=True)
                window['-sound-'].Update(image_data=icons['soundOn'], image_size=(25,25), image_subsample=18)
        if event == '-start-':
            if window['-logged_user-'].get() != '--':
                logged_user = window['-logged_user-'].get()
                user_data = db.search(User.nome == logged_user)[0]

                thread = threading.Thread(target=coletar, args=(window,values,user_data))
                thread.start()
        if event == '-graph-':
            modal = GraphModal(conf_mat)
            modal.open_window()

    window.close()

if __name__ == '__main__':
    main()