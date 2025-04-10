import FreeSimpleGUI as sg
from tinydb import TinyDB, Query
from datetime import datetime
from os import path
import csv

##TODO
# Adicionar opção de tempo de descanso RANDOM MI (com std e mean)
# Adicionar parâmetro de coleta na validação

class NewUser():

    def __init__(self):
        
        electrodes1020 = [
    "Fp1", "Fp2",  # Frontopolar
    "F3", "F4",    # Frontal
    "C3", "C4",    # Central
    "P3", "P4",    # Parietal
    "O1", "O2",    # Occipital
    "F7", "F8",    # Frontotemporal (lateral frontal)
    "T3", "T4",    # Temporal
    "T5", "T6",    # Temporoparietal (lateral temporal)
    "Fz",          # Frontal central
    "Cz",          # Central (vertex)
    "Pz",          # Parietal central
    "A1", "A2"     # Earlobe or mastoid (referência)
]

        self._layout = [[sg.Text('Nome')],
                    [sg.Input('', key='-name-', size=(30,1))],
                    [sg.Text('Descrição breve')],
                    [sg.Multiline('', no_scrollbar=True, do_not_clear=True, size=(30,4), key="-desc-")],
                    [   sg.Column([[sg.Text('Resolução')], [sg.Combo(['10-20'], key='-resolution-', enable_events=True)]]),
                        sg.Column([[sg.Text('Nº Canais')],[sg.Combo([i for i in range(1,22)], key='-number_channels-', disabled=True)]]),
                        sg.Column([[sg.Text('Canais')],[sg.Listbox(electrodes1020, key='-electrodes-', size=(4,4), select_mode='multiple')]])],
                    [sg.Text('Parametros de Coleta')],
                    [sg.Text('Runs'), sg.Text('Rest'), sg.Text('MI')],
                    [sg.Input('', key='-runs-', enable_events=True, size=(4,1)),
                    sg.Input('', key='-rest_time-', enable_events=True, size=(4,1)),
                    sg.Input('', key='-mi_time-', enable_events=True, size=(4,1))],
                    [sg.Text('', text_color='red', font=('Arial', 8), key='-warning-',visible=False)],
                    [sg.Button('Pronto', key='-ok-')],
                    ]

        self._db_path = path.join('users', 'users.json')
        self._db = TinyDB(self._db_path)
        
    def open_window(self):
        window = sg.Window("Criar Perfil de Coleta", self._layout, modal=True)

        while True:
            event, values = window.read()
            
            self._integer_input(event, values, window)
            if event == sg.WIN_CLOSED:
                window.close()
                return ()
            if event == '-resolution-' and values['-resolution-'] == '10-20':
                window['-number_channels-'].update(disabled=False)
            if event == '-ok-':
                validation = self._validate_values(values)
                if validation != '':
                    window['-warning-'].Update(validation, visible=True)
                else:
                    #cria o perfil no banco local
                    self._db.insert({'nome':values['-name-'],
                                    'desc':values['-desc-'],
                                    'resolution':values['-resolution-'],
                                    'number_channels':values['-number_channels-'],
                                    'channels':values['-electrodes-'],
                                    'numero_runs':int(values['-runs-']),
                                    'tempo_descanso':int(values['-rest_time-']),
                                    'tempo_imagetica':int(values['-mi_time-']),
                                    'data_criacao':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                    'ultimo_acesso':datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
                    #cria a planilha com dados de coleta
                    user_path = path.join('users', f'{values["-name-"]}.csv')
                    with open(user_path, 'w', newline='') as csvfile:
                        fieldnames = ['timestamp'] + values['-electrodes-'] + ['events'] # Adicione os nomes dos canais
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                    window.close()
                    return (values['-name-'], datetime.now().strftime("%d/%m/%Y"))
                    

    def _validate_values(self, values):
        
        if not values['-name-'].isalnum():
            return 'Nome inválido'
 
        if values['-desc-'].isspace():
            return 'Descrição inválida'

        #procura se já há usuário com esse nome
        User = Query()
        result = self._db.search(User.nome == values['-name-'])
        if len(result) != 0:
            return 'Nome já existe na base de dados'

        if not values['-name-']:
            return 'Ao menos um canal deve ser utilizado'

        if len(values['-electrodes-']) != int(values['-number_channels-']):
            return 'Número de eletrodos selecionados incorreto'

        return ''

    def _integer_input(self, event, values, window):
        if event in ['-channels-', '-runs-', '-rest_time-', '-mi_time-'] and len(values[event]) and values[event][-1] not in ('0123456789'):
            # delete last char from input
            window[event].update(values[event][:-1])

class ChangeUser:
    
    def __init__(self):

        self._db_path = path.join('users', 'users.json')
        self._db = TinyDB(self._db_path)

        users = self._db.all()

        usernames = [user['nome'] for user in users]

        self._layout = [[sg.Text('Escolha um Perfil: '),
                         sg.Combo(usernames, key='-user-')            
                        ],
                        [sg.Button('Pronto', key='-ok-')]
                        ]

    def open_window(self):

        window = sg.Window("Mudar Perfil de Coleta", self._layout, modal=True)

        while True:
            event, values = window.read()

            if event == sg.WIN_CLOSED:
                window.close()
                return ()
            if event == '-ok-':
                User = Query()
                usuario = self._db.search(User.nome == values['-user-'])[0]
                result = self._db.update({'ultimo_acesso': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}, User.nome == values['-user-'])
                window.close()
                return (values['-user-'], usuario['ultimo_acesso'])