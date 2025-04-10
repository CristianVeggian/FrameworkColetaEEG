import FreeSimpleGUI as sg
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class GraphModal:

    def __init__(self, conf_mat):

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot()
        plt.savefig('latest_conf_mat')
        self._layout = [[sg.Image('latest_conf_mat.png')]] 

    def open_window(self):
        window = sg.Window("Gráficos", self._layout, modal=True)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == '-cancel-':
                break
            
        window.close()
        return