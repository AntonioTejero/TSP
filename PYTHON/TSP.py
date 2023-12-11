import sys

import matplotlib as mpl
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import QtGui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("TSP")
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        
        mainLayout = QtWidgets.QVBoxLayout(self._main)
        
        titleLabel = QtWidgets.QLabel("The Traveling Salesperson Problem (TSP)")
        titleLabel.setFont(QtGui.QFont('Arial', 24))
        mainLayout.addWidget(titleLabel)
        #mainLayout.addWidget(leftLayout)
        #mainLayout.addWidget(rightLayout)
        
        self.numberOfCitiesSpinBox = QtWidgets.QSpinBox()
        self.numberOfCitiesSpinBox.setMinimum(4)
        mainLayout.addWidget(self.numberOfCitiesSpinBox)
        generateButton = QtWidgets.QPushButton("Generate!")
        generateButton.clicked.connect(self.generateButtonClicked)
        mainLayout.addWidget(generateButton)

        static_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        mainLayout.addWidget(static_canvas)
        mainLayout.addWidget(NavigationToolbar2QT(static_canvas, self))
        
        self._static_canvas = static_canvas
        self._static_ax = static_canvas.figure.subplots()
        self._static_ax.set_aspect('equal')
        self._static_ax.set(xlim=(0,1), ylim=(0,1))
        self._static_ax.set_title('Map of cities')
        self._static_ax.set_xlabel('X position (A.U.)')
        self._static_ax.set_ylabel('Y position (A.U.)')
        self.scatter = self._static_ax.scatter([], [], c='r')
        self.plot, = self._static_ax.plot([], [], c='grey')
        
    def generateButtonClicked(self):
        numberOfCities = self.numberOfCitiesSpinBox.value()
        cityXpos = np.random.uniform(0,1,numberOfCities)
        cityYpos = np.random.uniform(0,1,numberOfCities)
        distance = np.zeros( (numberOfCities, numberOfCities) )
        for i in range(numberOfCities):
          for j in range(i+1,numberOfCities):
            distance[i][j] = np.sqrt((cityXpos[i]-cityXpos[j])**2+(cityYpos[i]-cityYpos[j])**2)
            distance[j][i] = distance[i][j]
        
        self.plot.set_xdata(cityXpos[list(range(numberOfCities))+[0]])
        self.plot.set_ydata(cityYpos[list(range(numberOfCities))+[0]])
        self.scatter.set_offsets(np.column_stack((cityXpos, cityYpos)))
        for child in self._static_ax.get_children():
            if isinstance(child, mpl.text.Annotation):
                child.remove()
        for i in range(numberOfCities):
            self._static_ax.annotate(i+1, xy=(cityXpos[i], cityYpos[i]), fontsize=15, xytext=(5,5), textcoords='offset points')
        
        self.scatter.figure.canvas.draw()

qapp = QtWidgets.QApplication(sys.argv)
appWindow = MainWindow()
appWindow.show()
appWindow.activateWindow()
appWindow.raise_()
qapp.exec()
