import FreeSimpleGUI as sg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA_modal:

    """
        Janela modal para a configuração de um novo LDA
         priors=None
         covariance_estimator=None
    """

    def __init__(self):
        self._names_column = [  [sg.Text('solver')],
                                [sg.Text('shrinkage')],
                                [sg.Text('n_components')],
                                [sg.Text('store_covariance')],
                                [sg.Text('tol')],
                                ]
        self._entries_column = [[sg.Combo(['svd', 'lsqr', 'eigen'], default_value='svd', key='-LDA_solver-', readonly=True, size=(20,1))],
                                [sg.Combo(['-','auto','float'], default_value='-', key='-LDA_shrinkage-', readonly=True, size=(8,1), enable_events=True),
                                sg.Spin([i/10000 for i in range(0,10000)], initial_value=0.000, key='-LDA_shrinkage_float-', size=(8,1), disabled=True)],
                                [sg.Spin([i for i in range(0,100)], initial_value='-', key='-LDA_n_comp-', size=(20,1))],
                                [sg.Combo(['True','False'], default_value='True', readonly=True, key='-LDA_store_covariance-', size=(20,1))],
                                [sg.Spin([i/10000 for i in range(0,10000)], initial_value=0.0001, key='-LDA_tol-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-LDA_OK-')]]

    def open_window(self):
        window = sg.Window("Parâmetros LDA", self._layout, modal=True)

        while True:
            event, values = window.read()
            
            if event == '-LDA_shrinkage-':
                if values['-LDA_shrinkage-'] == 'float':
                    window['-LDA_shrinkage_float-'].Update(disabled=False)
                else:
                    window['-LDA_shrinkage_float-'].Update(disabled=True)
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
            if event == '-LDA_OK-':
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

        if values['-LDA_shrinkage-'] == 'float' and type(values['-LDA_shrinkage_float-']) != float:
            try:
                values['-LDA_shrinkage_float-'] = float(values['-LDA_shrinkage_float-'])
            except:
                return 'Shrinkage float precisa ser um número real.'

        if values['-LDA_shrinkage_float-'] < 0 or values['-LDA_shrinkage_float-'] > 1:
            return 'Shrinkage deve estar entre 0 e 1.'

        if values['-LDA_n_comp-'] == '-':
            values['-LDA_n_comp-'] = None
        elif type(values['-LDA_n_comp-']) != int:
            return 'Número de componentes deve ser número inteiro ou - (None)'
            
        if type(values['-LDA_tol-']) != float:
            return 'Tolerance precisa ser um número real.'

        return ''

    def build_method(self, values):
        
        if values['-LDA_shrinkage-'] == '-':
            values['-LDA_shrinkage-'] = None
        elif values['-LDA_shrinkage-'] == 'float':
            values['-LDA_shrinkage-'] = float(values['-LDA_shrinkage_float-'])

        if values['-LDA_store_covariance-'] == 'True':
            values['-LDA_store_covariance-'] = True
        else:
            values['-LDA_store_covariance-'] = False

        lda = LinearDiscriminantAnalysis(solver=values['-LDA_solver-'],
                                         shrinkage=values['-LDA_shrinkage-'],
                                         n_components=values['-LDA_n_comp-'],
                                         store_covariance=values['-LDA_store_covariance-'],
                                         tol=values['-LDA_tol-'],
        )

        return {'method': ("LDA", lda), 'args': values}
        
        