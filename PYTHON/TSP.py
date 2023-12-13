import sys

import matplotlib as mpl
import numpy as np
import time
import copy

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import QtGui
from matplotlib.backends.qt_compat import QtCore
from itertools import permutations


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("TSP")
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        
        mainLayout = QtWidgets.QVBoxLayout(self._main)
        centralLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        rightLayout = QtWidgets.QVBoxLayout()
        
        titleLabel = QtWidgets.QLabel("The Traveling Salesperson Problem (TSP)")
        titleLabel.setFont(QtGui.QFont('Arial', 24))
        titleLabel.setMaximumSize(QtCore.QSize(2000,35))
        mainLayout.addWidget(titleLabel)
        mainLayout.addLayout(centralLayout)
        centralLayout.addLayout(leftLayout)
        centralLayout.addLayout(rightLayout)
        
        initializationGroup = QtWidgets.QGroupBox("Initialization:")
        initializationGroup.setFixedSize(QtCore.QSize(230,90))
        leftLayout.addWidget(initializationGroup)
        initializationLayout = QtWidgets.QVBoxLayout(initializationGroup)
        numberOfCitiesLayout = QtWidgets.QHBoxLayout()
        initializationLayout.addLayout(numberOfCitiesLayout)
        numberOfCitiesLabel = QtWidgets.QLabel("Number of cities: ")
        numberOfCitiesLayout.addWidget(numberOfCitiesLabel)
        self.numberOfCitiesSpinBox = QtWidgets.QSpinBox()
        self.numberOfCitiesSpinBox.setMinimum(4)
        self.numberOfCitiesSpinBox.setMaximum(999)
        numberOfCitiesLayout.addWidget(self.numberOfCitiesSpinBox)
        generateButton = QtWidgets.QPushButton("Generate!")
        generateButton.clicked.connect(self.generateButtonClicked)
        initializationLayout.addWidget(generateButton)
        self.generateButton = generateButton
        
        solverGroup = QtWidgets.QGroupBox("Solver:")
        solverGroup.setFixedSize(QtCore.QSize(230,110))
        leftLayout.addWidget(solverGroup)
        solverLayout = QtWidgets.QVBoxLayout(solverGroup)
        algorithmSelectionBox = QtWidgets.QComboBox()
        algorithmSelectionBox.addItem("Exact")
        algorithmSelectionBox.addItem("Nearest neighbour")
        algorithmSelectionBox.setEnabled(False)
        solverLayout.addWidget(algorithmSelectionBox)
        algorithmSelectionBox.currentTextChanged.connect(self.algorithmSelectedChanged)
        self.algorithmSelectionBox = algorithmSelectionBox
        liveSolverLayout = QtWidgets.QHBoxLayout()
        solverLayout.addLayout(liveSolverLayout)
        liveSolverLabel = QtWidgets.QLabel("Activate live solver:")
        liveSolverLayout.addWidget(liveSolverLabel)
        liveSolverCheckBox = QtWidgets.QCheckBox()
        liveSolverCheckBox.setChecked(True)
        liveSolverCheckBox.setEnabled(False)
        liveSolverLayout.addWidget(liveSolverCheckBox)
        self.liveSolverCheckBox = liveSolverCheckBox
        solveItButton = QtWidgets.QPushButton("Solve it!")
        solveItButton.setEnabled(False)
        solverLayout.addWidget(solveItButton)
        solveItButton.clicked.connect(self.solveItButtonClicked)
        self.solveItButton = solveItButton
        
        solutionsGroup = QtWidgets.QGroupBox("Solutions:")
        solutionsGroup.setMinimumSize(QtCore.QSize(230,0))
        solutionsGroup.setMaximumSize(QtCore.QSize(230,2000))
        leftLayout.addWidget(solutionsGroup)
        solutionsLayout = QtWidgets.QVBoxLayout(solutionsGroup)
        solutionsLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        solutionSelectionBox = QtWidgets.QComboBox()
        solutionSelectionBox.setEnabled(False)
        solutionSelectionBox.currentTextChanged.connect(self.solutionSelectedChanged)
        solutionsLayout.addWidget(solutionSelectionBox)
        solutionRouteLabel = QtWidgets.QLabel("Route: ")
        solutionRouteLabel.setWordWrap(True)
        solutionsLayout.addWidget(solutionRouteLabel)
        solutionDistanceLabel = QtWidgets.QLabel("Distance: ")
        solutionsLayout.addWidget(solutionDistanceLabel)
        solutionExecutionTimeLabel = QtWidgets.QLabel("Execution time (s): ")
        solutionsLayout.addWidget(solutionExecutionTimeLabel)
        self.solutionSelectionBox = solutionSelectionBox
        self.solutionRouteLabel = solutionRouteLabel
        self.solutionDistanceLabel = solutionDistanceLabel
        self.solutionExecutionTimeLabel = solutionExecutionTimeLabel

        graphisGroup = QtWidgets.QGroupBox("Graphs:")
        rightLayout.addWidget(graphisGroup)
        graphicsLayout = QtWidgets.QVBoxLayout(graphisGroup)
        static_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        graphicsLayout.addWidget(static_canvas)
        graphicsLayout.addWidget(NavigationToolbar2QT(static_canvas, self))
        
        self._static_canvas = static_canvas
        self._static_ax = static_canvas.figure.subplots()
        self._static_canvas.setMinimumSize(QtCore.QSize(500,500))
        self._static_ax.set_aspect('equal')
        self._static_ax.set(xlim=(0,1), ylim=(0,1))
        self._static_ax.set_title('Map of cities')
        self._static_ax.set_xlabel('X position (A.U.)')
        self._static_ax.set_ylabel('Y position (A.U.)')
        self.scatter = self._static_ax.scatter([], [], c='r')
        self.plot, = self._static_ax.plot([], [], c='grey')
        
        self.solutionsExactRoute = []
        self.solutionsExactDistance = []
        self.solutionsExactExecutionTime = []
        self.solutionsClosestNeighbourRoute = []
        self.solutionsClosestNeighbourDistance = []
        self.solutionsClosestNeighbourExecutionTime = []
        
    def generateButtonClicked(self):
        
        numberOfCities = self.numberOfCitiesSpinBox.value()
        cityXpos = np.random.uniform(0,1,numberOfCities)
        cityYpos = np.random.uniform(0,1,numberOfCities)
        distance = np.zeros( (numberOfCities, numberOfCities) )
        for i in range(numberOfCities):
            distance[i][i] = float('inf')
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
        QtCore.QCoreApplication.processEvents()
        
        self.algorithmSelectionBox.setEnabled(True)
        if numberOfCities<=10:
            self.solveItButton.setEnabled(True)
        elif self.algorithmSelectionBox.currentText() == "Nearest neighbour":
            self.solveItButton.setEnabled(True)
        else:
            self.solveItButton.setEnabled(False)
        self.liveSolverCheckBox.setEnabled(True)
            
        self.solutionsExactRoute = []
        self.solutionsExactDistance = []
        self.solutionsExactExecutionTime = []
        self.solutionsClosestNeighbourRoute = []
        self.solutionsClosestNeighbourDistance = []
        self.solutionsClosestNeighbourExecutionTime = []
        self.solutionSelectionBox.clear()
        self.solutionSelectionBox.setEnabled(False)
        self.solutionRouteLabel.setText("Route: ")
        self.solutionDistanceLabel.setText("Distance: ")
        self.solutionExecutionTimeLabel.setText("Execution time (s): ")
            
        self.numberOfCities = numberOfCities
        self.cityXpos = cityXpos
        self.cityYpos = cityYpos
        self.distance = distance
        
    def algorithmSelectedChanged(self):
        
        if hasattr(self, 'numberOfCities'):
            if self.algorithmSelectionBox.currentText() == "Exact" and self.numberOfCities<=10 and self.solutionsExactRoute == []:
                self.solveItButton.setEnabled(True)
            elif self.algorithmSelectionBox.currentText() == "Nearest neighbour" and self.solutionsClosestNeighbourRoute == []:
                self.solveItButton.setEnabled(True)
            else:
                self.solveItButton.setEnabled(False)
        else:
            return
    
    def solveItButtonClicked(self):
        
        print(self.algorithmSelectionBox.currentText())
        if self.algorithmSelectionBox.currentText() == "Exact":
            self.solveExact()
        elif self.algorithmSelectionBox.currentText() == "Nearest neighbour":
            self.solveNearestNeighbour()
        else:
            print("No luck...")
            
    def solveExact(self):
        
        self.numberOfCitiesSpinBox.setEnabled(False)
        self.generateButton.setEnabled(False)
        self.algorithmSelectionBox.setEnabled(False)
        self.solveItButton.setEnabled(False)
        self.solutionSelectionBox.setEnabled(False)
        
        startTime = time.time()
        
        routes = permutations(range(1, self.numberOfCities))
        bestRoute = []
        bestRouteDistance = float('inf')
        for route in routes:
            route = [0] + list(route) + [0]
            routeTotalDistance = 0
            for idx in range(self.numberOfCities):
                i = route[idx]
                j = route[idx+1]
                routeTotalDistance = routeTotalDistance + self.distance[i][j]
            if routeTotalDistance < bestRouteDistance:
                bestRoute = route
                bestRouteDistance = routeTotalDistance
            print("Testing route: ", route)
            if self.liveSolverCheckBox.isChecked():
                self.plot.set_xdata(self.cityXpos[route])
                self.plot.set_ydata(self.cityYpos[route])
                self.scatter.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()
                
            
        print("Best route: ", bestRoute, "(", bestRouteDistance, ")")
        self.plot.set_xdata(self.cityXpos[bestRoute])
        self.plot.set_ydata(self.cityYpos[bestRoute])
        self.scatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
        finishTime = time.time()
        
        self.solutionsExactRoute = bestRoute
        self.solutionsExactDistance = bestRouteDistance
        self.solutionsExactExecutionTime = finishTime-startTime
        self.solutionSelectionBox.addItem("Exact")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in bestRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(bestRouteDistance))
        self.solutionExecutionTimeLabel.setText("Execution time (s): " + str(finishTime-startTime))
        
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        
        self
        
    def solveNearestNeighbour(self):
        
        self.numberOfCitiesSpinBox.setEnabled(False)
        self.generateButton.setEnabled(False)
        self.algorithmSelectionBox.setEnabled(False)
        self.solveItButton.setEnabled(False)
        self.solutionSelectionBox.setEnabled(False)
        
        startTime = time.time()
        
        bestRoute = []
        bestRouteDistance = float('inf')
        for initialCity in range(self.numberOfCities):
            route = [initialCity]
            routeTotalDistance = 0
            for idx in range(self.numberOfCities-1):
                possibleDistances = copy.copy(self.distance[route[-1]][:])
                for visitedCity in route:
                    possibleDistances[visitedCity] = float('inf')
                nextCity = np.argmin(possibleDistances)
                route.append(nextCity)
                routeTotalDistance = routeTotalDistance + possibleDistances[nextCity]
            route.append(initialCity)
            routeTotalDistance = routeTotalDistance + self.distance[route[-2]][route[-1]]
            print("Testing route: ", route)
            if self.liveSolverCheckBox.isChecked():
                self.plot.set_xdata(self.cityXpos[route])
                self.plot.set_ydata(self.cityYpos[route])
                self.scatter.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()
            if routeTotalDistance<bestRouteDistance:
                bestRoute = route
                bestRouteDistance = routeTotalDistance
            
        print("Best route: ", bestRoute, "(", bestRouteDistance, ")")
        self.plot.set_xdata(self.cityXpos[bestRoute])
        self.plot.set_ydata(self.cityYpos[bestRoute])
        self.scatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
        finishTime = time.time() 
        
        self.solutionsClosestNeighbourRoute = bestRoute
        self.solutionsClosestNeighbourDistance = bestRouteDistance
        self.solutionsClosestNeighbourExecutionTime = finishTime-startTime
        self.solutionSelectionBox.addItem("Closest neighbour")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in bestRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(bestRouteDistance))
        self.solutionExecutionTimeLabel.setText("Execution time (s): " + str(finishTime-startTime))
        
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        
    def solutionSelectedChanged(self):
        
        if self.solutionSelectionBox.currentText() == "Exact":
            route = self.solutionsExactRoute
            distance = self.solutionsExactDistance
            executionTime = self.solutionsExactExecutionTime
        elif self.solutionSelectionBox.currentText() == "Closest neighbour":
            route = self.solutionsClosestNeighbourRoute
            distance = self.solutionsClosestNeighbourDistance
            executionTime = self.solutionsClosestNeighbourExecutionTime
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in route]))
        self.solutionDistanceLabel.setText("Distance: " + str(distance))
        self.solutionExecutionTimeLabel.setText("Execution Time (s): " + str(executionTime))
        self.plot.set_xdata(self.cityXpos[route])
        self.plot.set_ydata(self.cityYpos[route])
        self.scatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
            
qapp = QtWidgets.QApplication(sys.argv)
appWindow = MainWindow()
appWindow.show()
appWindow.activateWindow()
appWindow.raise_()
qapp.exec()
