import FreeSimpleGUI as sg
from sklearn.neural_network import MLPClassifier

class MLP_modal:

    """
        Janela modal para a configuração de um novo SVM

    """

    def __init__(self):
        self._names_column = [  [sg.Text('Camadas Ocultas')],
                                [sg.Text('Função de Ativação')],
                                [sg.Text('Solver')],
                                [sg.Text('Alpha')],
                                #[sg.Text('Batch Size')]
                                ]
        self._entries_column = [[sg.Input('100', enable_events=True, key='-MLP_hidden_layers-', size=(20,1))],
                                [sg.Combo(['identity', 'logistic', 'tanh', 'relu'], readonly=True, default_value='relu', key='-MLP_activation-', size=(20,1))],
                                [sg.Combo(['lbfgs', 'sgd', 'adam'], readonly=True, default_value='adam', key='-MLP_solver-', size=(20,1))],
                                [sg.Input('0.001', enable_events=True, key='-MLP_alpha-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-ok-')]]

    def open_window(self):
        window = sg.Window("Parâmetros MLP", self._layout, modal=True)

        while True:
            event, values = window.read()
            self._integer_input(event, values, window)
            self._float_input(event, values, window)
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
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
        if event in ['-MLP_hidden_layers-'] and len(values[event]) and values[event][-1] not in ('0123456789'):
            # delete last char from input
            window[event].update(values[event][:-1])

    def _float_input(self, event, values, window):
        if event in ['-MLP_alpha-'] and len(values[event]) and values[event][-1] not in ('0123456789.'):
            # delete last char from input
            window[event].update(values[event][:-1])


    def build_method(self, values):
        
        values['-MLP_hidden_layers-'] = int(values['-MLP_hidden_layers-'])
        values['-MLP_alpha-'] = float(values['-MLP_alpha-'])

        mlp = MLPClassifier(hidden_layer_sizes=(values['-MLP_hidden_layers-'],),
                            activation=values['-MLP_activation-'],
                            solver=values['-MLP_solver-'],
                            alpha=values['-MLP_alpha-'],
                            max_iter=1000
        )

        return {'method': ("MLP", mlp), 'args': values}
        
        