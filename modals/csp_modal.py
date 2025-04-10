import FreeSimpleGUI as sg
from mne.decoding import CSP
import textwrap

wrapper = textwrap.TextWrapper(width=50)

class CSP_modal:

    """
        Janela modal para a configuração de um novo CSP
        cov_method_params=None
        rank=None
        component_order='mutual_info'
    """

    def __init__(self):
        self._names_column = [  [sg.Text('Número de Componentes', tooltip=wrapper.fill("Número de componentes para decompor os sinais de EEG."))],
                                [sg.Text('Regularização para Covariância', tooltip=wrapper.fill("Regularização para a estimação da covariância. Se não for \" - \", permite a regularização para a estimação da covariância. Se float (entre 0 e 1), é usado encolhimento. Para outros valores, será passado como método para o pipeline"))],
                                [sg.Text('Transformação Logarítmica')],
                                [sg.Text('Estimativa de Covariância', tooltip=wrapper.fill("Se 'concat', as matrizes de covariância são estimadas em épocas concatenadas para cada classe. Se 'epoch', as matrizes de covariância são estimadas em cada época separadamente e depois são tiradas médias para cada classe"))],
                                [sg.Text('Resultado de Saída', tooltip=wrapper.fill("Se 'average_power', retornará a potência média de cada filtro espacial. Se 'csp_space', retornará os dados no espaço comum"))],
                                [sg.Text('Normalização de Traço', tooltip=wrapper.fill("A normalização do traço é uma etapa do algoritmo CSP original para eliminar variações de magnitude no EEG entre indivíduos. Não é aplicada em trabalhos mais recentes e pode ter um impacto negativo na ordem do padrão."))],
                                ]
        self._entries_column = [[sg.Spin([i for i in range(1,100)], initial_value=4, key='-CSP_n_comp-', size=(20,1))],
                                [sg.Combo(['-','shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis'], default_value='-', key='-CSP_reg-', size=(20,1))],
                                [sg.Combo(['-','True','False'], default_value='-', key='-CSP_log-', size=(20,1))],
                                [sg.Combo(['concat', 'epoch'], default_value='concat', key='-CSP_cov_est-', size=(20,1))],
                                [sg.Combo(['average_power', 'csp_space'], default_value='average_power', key='-CSP_transform_into-', size=(20,1))],
                                [sg.Combo(['True','False'], default_value='True', key='-CSP_norm_trace-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-CSP_OK-')]]

    def open_window(self):
        window = sg.Window("Parâmetros CSP", self._layout, modal=True)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
            if event == '-CSP_OK-':
                validation = self._validate_values(values)
                if validation != '':
                    window['-warning-'].Update(validation, visible=True)
                else:
                    window.close()
                    return self.build_method(values)                    
        
        window.close()

    def _validate_values(self, values):

        """
        Validação dos valores dentro da modal, garante que 
        os dados inseridos seguem o padrão necessário da
        aplicação. 
        """

        if type(values['-CSP_n_comp-']) != int:
            return 'Número de componentes deve ser número inteiro!'
        if values['-CSP_reg-'] not in ['-','shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']:
            return 'Regularização inválida'
        if values['-CSP_log-'] not in ['-','True','False']:
            return 'Log inválido'
        if values['-CSP_cov_est-'] not in ['concat', 'epoch']:
            return 'Cov est inválido'
        if values['-CSP_transform_into-'] not in ['average_power', 'csp_space']:
            return 'Transform into inválido'
        if values['-CSP_norm_trace-'] not in ['False', 'True']:
            return 'Norm Trace inválida'
        return ''

    def build_method(self, values):
        
        if values['-CSP_reg-'] == '-':
            values['-CSP_reg-'] = None

        if values['-CSP_log-'] == '-':
            values['-CSP_log-'] = None
        elif values['-CSP_log-'] == 'True':
            values['-CSP_log-'] = True
        else:
            values['-CSP_log-'] = False

        if values['-CSP_norm_trace-'] == 'True':
            values['-CSP_norm_trace-'] = True
        else:
            values['-CSP_norm_trace-'] = False
        
        csp = CSP(  n_components=values['-CSP_n_comp-'],
                    reg=values['-CSP_reg-'],
                    log=values['-CSP_log-'],
                    cov_est=values['-CSP_cov_est-'],
                    transform_into=values['-CSP_transform_into-'],
                    norm_trace=values['-CSP_norm_trace-'])

        return {'method': ("CSP", csp), 'args': values}
        
        