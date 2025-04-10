import FreeSimpleGUI as sg
from sklearn.svm import SVC

class SVM_modal:

    """
        Janela modal para a configuração de um novo SVM

    """

    def __init__(self):
        self._names_column = [  [sg.Text('Parâmetro de Regularização')],
                                [sg.Text('Kernel')],
                                [sg.Text('Grau', key='-degree_text-')],
                                [sg.Text('Gama', key='-gamma_text-')],
                                [sg.Text('Coeficiente 0', key='-coef0_text-')],
                                [sg.Text('Heurística de Redução')],
                                [sg.Text('Estimativa de Probabilidade')],
                                [sg.Text('Tolerância')],
                                [sg.Text('Cache do Kernel')],
                                [sg.Text('Máximo de Iterações')]
                                ]
        self._entries_column = [[sg.Spin([i/10000 for i in range(0,10000)], initial_value=1.000, key='-SVM_reg-', size=(20,1))],
                                [sg.Combo(['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], readonly=True, default_value='rbf', key='-SVM_kernel-', size=(20,1), enable_events=True)],
                                [sg.Spin([i for i in range(0,10)], initial_value=3, key='-SVM_degree-', size=(20,1), disabled=True)],
                                [sg.Combo(['scale', 'auto', 'float'], disabled=False, readonly=True, default_value='scale', key='-SVM_gamma-', size=(8,1), enable_events=True),
                                 sg.Spin([i/10000 for i in range(0,10000)], disabled=True, initial_value=0.000, key='-SVM_gamma_float-', size=(8,1))],
                                [sg.Spin([i/10000 for i in range(0,10000)], disabled=True, initial_value=0.000, key='-SVM_coef0-', size=(20,1))],
                                [sg.Combo(['True','False'], readonly=True, default_value='True', key='-SVM_shrinking-', size=(20,1))],
                                [sg.Combo(['True','False'], readonly=True, default_value='False', key='-SVM_probability-', size=(20,1))],
                                [sg.Spin([i/1000 for i in range(0,1000)], initial_value=0.001, key='-SVM_tol-', size=(20,1))],
                                [sg.Spin([i for i in range(0,1048576)], initial_value=200.000, key='-SVM_cache_size-', size=(20,1))],
                                [sg.Spin([i for i in range(0,1000)], initial_value=-1, key='-SVM_max_iter-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-ok-')]]

    def open_window(self):
        window = sg.Window("Parâmetros SVM", self._layout, modal=True)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
            if event == '-SVM_kernel-':
                if values['-SVM_kernel-'] == 'poly':
                    window['-SVM_gamma-'].Update(disabled=False)
                    window['-SVM_gamma_float-'].Update(disabled=False)
                    window['-SVM_degree-'].Update(disabled=False)
                    window['-SVM_coef0-'].Update(disabled=False)
                elif values['-SVM_kernel-'] == 'rbf':
                    window['-SVM_gamma-'].Update(disabled=False)
                    window['-SVM_gamma_float-'].Update(disabled=False)
                    window['-SVM_degree-'].Update(disabled=True)
                    window['-SVM_coef0-'].Update(disabled=True)
                elif values['-SVM_kernel-'] == 'sigmoid':
                    window['-SVM_gamma-'].Update(disabled=False)
                    window['-SVM_gamma_float-'].Update(disabled=False)
                    window['-SVM_degree-'].Update(disabled=True)
                    window['-SVM_coef0-'].Update(disabled=False)
                else:
                    window['-SVM_gamma-'].Update(disabled=True)
                    window['-SVM_gamma_float-'].Update(disabled=True)
                    window['-SVM_degree-'].Update(disabled=True)
                    window['-SVM_coef0-'].Update(disabled=True)
            if values['-SVM_gamma-'] == 'float':
                window['-SVM_gamma_float-'].Update(disabled=False)
            else:
                window['-SVM_gamma_float-'].Update(disabled=True)
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

        if values['-SVM_gamma-'] == 'float' and type(values['-SVM_gamma_float-']) != float:
            try:
                values['-SVM_gamma_float-'] = float(values['-SVM_gamma_float-'])
            except:
                return 'O valor de gamma precisa ser um número real.'

        try:
            values['-SVM_reg-'] = float(values['-SVM_reg-'])
        except:
            return 'Parâmetro de regularização deve ser número real positivo.'

        if values['-SVM_reg-'] < 0:
            return 'Parâmetro de regularização deve ser número positivo.'

        try:
            values['-SVM_degree-'] = int(values['-SVM_degree-'])
        except:
            return 'Grau deve ser número inteiro'

        if values['-SVM_degree-'] < 0:
            return 'Grau deve ser número positivo.'

        try:
            values['-SVM_tol-'] = float(values['-SVM_tol-'])
        except:
            return 'Valor de tolerância deve ser número real'

        if values['-SVM_tol-'] < 0:
            return 'Valor de tolerância deve ser número positivo.'

        try:
            values['-SVM_max_iter-'] = int(values['-SVM_max_iter-'])
        except:
            return 'Máximo de iterações deve ser número inteiro'
            
        if values['-SVM_max_iter-'] < -1:
            return 'Máximo de iterações deve ser número positivo.'
        
        try:
            values['-SVM_cache_size-'] = float(values['-SVM_cache_size-'])
        except:
            return 'Tamanho de Cache deve ser número inteiro'

        if values['-SVM_cache_size-'] < 0:
            return 'Tamanho de Cache deve ser número positivo.'

        return ''

    def build_method(self, values):
        
        if values['-SVM_gamma-'] == 'float':
            values['-SVM_gamma-'] = float(values['-SVM_gamma_float-'])

        if values['-SVM_shrinking-'] == 'True':
            values['-SVM_shrinking-'] = True
        else:
            values['-SVM_shrinking-'] = False

        if values['-SVM_probability-'] == 'True':
            values['-SVM_probability-'] = True
        else:
            values['-SVM_probability-'] = False

        svm = SVC(C=values['-SVM_reg-'],
                kernel=values['-SVM_kernel-'],
                degree=values['-SVM_degree-'],
                gamma=values['-SVM_gamma-'],
                coef0=float(values['-SVM_coef0-']),
                shrinking=values['-SVM_shrinking-'],
                probability=values['-SVM_probability-'],
                tol=values['-SVM_tol-'],
                cache_size=values['-SVM_cache_size-'],
                max_iter=values['-SVM_max_iter-']
                )

        return {'method': ("SVM", svm), 'args': values}
        
        