import FreeSimpleGUI as sg
from sklearn.neighbors import KNeighborsClassifier

class KNN_modal:

    """
        Janela modal para a configuração de um novo KNN

    """

    def __init__(self):
        self._names_column = [  [sg.Text('Número de Vizinhos')],
                                [sg.Text('Pesos')],
                                [sg.Text('Algoritmo')],
                                [sg.Text('Tamanho da Folha')],
                                #[sg.Text('Batch Size')]
                                ]
        self._entries_column = [[sg.Input('5', enable_events=True, key='-KNN_neighbours-', size=(20,1))],
                                [sg.Combo(['uniform', 'distance'], readonly=True, default_value='uniform', key='-KNN_weights-', size=(20,1))],
                                [sg.Combo(['auto', 'ball_tree', 'kd_tree', 'brute'], readonly=True, default_value='auto', key='-KNN_algorithm-', size=(20,1), enable_events=True)],
                                [sg.Input('30', enable_events=True, key='-KNN_leaf_size-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-ok-')]]

    def open_window(self):
        window = sg.Window("Parâmetros KNN", self._layout, modal=True)

        while True:
            event, values = window.read()
            self._integer_input(event, values, window)
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
            if event == '-KNN_algorithm-':
                if values['-KNN_algorithm-'] in ['ball_tree', 'kd_tree']:
                    window['-KNN_leaf_size-'].Update(disabled=False)
                else:
                    window['-KNN_leaf_size-'].Update(disabled=True)
            if event == '-ok-':
                validation = self._validate_values(values)
                if validation != '':
                    window['-warning-'].Update(validation, visible=True)
                else:
                    window.close()
                    return self.build_method(values)

    def _validate_values(self, values):

        """
        Validação dos valores dentro da modal, garante que 
        os dados inseridos seguem o padrão necessário da
        aplicação. 
        """

        return ''

    def _integer_input(self, event, values, window):
        if event in ['-KNN_neighbours-', '-KNN_leaf_size-'] and len(values[event]) and values[event][-1] not in ('0123456789'):
            # delete last char from input
            window[event].update(values[event][:-1])

    def build_method(self, values):
        
        values['-KNN_neighbours-'] = int(values['-KNN_neighbours-'])
        values['-KNN_leaf_size-'] = int(values['-KNN_leaf_size-'])

        knn = KNeighborsClassifier(n_neighbors=values['-KNN_neighbours-'],
                                   weights=values['-KNN_weights-'],
                                   algorithm=values['-KNN_algorithm-'],
                                   leaf_size=values['-KNN_leaf_size-']
        )

        return {'method': ("KNN", knn), 'args': values}
        
        