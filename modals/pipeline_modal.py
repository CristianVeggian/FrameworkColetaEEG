import FreeSimpleGUI as sg
import re

class Pipeline_modal:

    """

    """

    def __init__(self, methods, args):

        self._layout = [[sg.Text('PIPELINE DE PROCESSAMENTO', justification='center', font=('Helvetica', 20))]] 

        position = 0
        for method, arg in zip(methods, args):
            self._layout.append([sg.Text(method[0], justification='center', size=(50,1)),
                                sg.Button("Remover", key=f'-remove_{position}-', button_color=('white', 'red')),
                                sg.Button("Manter", key=f'-keep_{position}-', button_color=('black', 'yellow'), visible=False)])
            position += 1
            colunaNomes = []
            colunaValor = []
            for name, value in arg.items():
                colunaNomes.append([sg.Text(name.strip('-').replace('_', ' '))])
                colunaValor.append([sg.Text(value)])
            self._layout.append([sg.Column(colunaNomes), sg.Column(colunaValor)])
            self._layout.append([sg.HorizontalSeparator()])

    def open_window(self):
        window = sg.Window("Pipeline de Processamento", self._layout, modal=True)

        remove = list()

        while True:
            event, values = window.read()
            
            if event == sg.WIN_CLOSED or event == '-cancel-':
                break
            if '-remove' in event:
                numero = re.search(r'\d+', event).group()
                position = int(numero)
                remove.append(position)
                window[event].Update(visible=False)
                window[f'-keep_{position}-'].Update(visible=True)
            if '-keep' in event:
                numero = re.search(r'\d+', event).group()
                position = int(numero)
                remove.remove(position)
                window[event].Update(visible=False)
                window[f'-remove_{position}-'].Update(visible=True)
            
        window.close()
        return remove