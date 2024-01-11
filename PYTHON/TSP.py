import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
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
        initializationGroup.setFixedSize(QtCore.QSize(220,90))
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
        solverGroup.setFixedSize(QtCore.QSize(220,300))
        leftLayout.addWidget(solverGroup)
        solverLayout = QtWidgets.QVBoxLayout(solverGroup)
        algorithmSelectionBox = QtWidgets.QComboBox()
        algorithmSelectionBox.addItem("Exact")
        algorithmSelectionBox.addItem("Closest neighbour")
        algorithmSelectionBox.addItem("Simulated annealing")
        algorithmSelectionBox.addItem("Genetic algorithm")
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
        solverConfLayout = QtWidgets.QStackedLayout()
        solverLayout.addLayout(solverConfLayout)
        solverConfWidgetDummy = QtWidgets.QWidget()
        solverConfLayout.addWidget(solverConfWidgetDummy)
        solverConfWidgetSimulatedAnnealing = QtWidgets.QWidget()
        solverConfLayout.addWidget(solverConfWidgetSimulatedAnnealing)
        solverConfLayoutSimulatedAnnealing = QtWidgets.QVBoxLayout(solverConfWidgetSimulatedAnnealing)
        solverConfLayoutSimulatedAnnealing.setContentsMargins(0,11,11,11)
        initialTemperatureLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(initialTemperatureLayout)
        initialTemperatureLabel = QtWidgets.QLabel("Initial temperature:")
        initialTemperatureLayout.addWidget(initialTemperatureLabel)
        initialTemperatureSpinBox = QtWidgets.QDoubleSpinBox()
        initialTemperatureSpinBox.setRange(0.1,100)
        initialTemperatureSpinBox.setSingleStep(0.1)
        initialTemperatureSpinBox.setDecimals(1)
        initialTemperatureSpinBox.setValue(1)
        initialTemperatureLayout.addWidget(initialTemperatureSpinBox)
        self.initialTemperatureSpinBox = initialTemperatureSpinBox
        thermalizingIterationsLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(thermalizingIterationsLayout)
        thermalizingIterationsLabel = QtWidgets.QLabel("Thermalizing iterations:")
        thermalizingIterationsLayout.addWidget(thermalizingIterationsLabel)
        thermalizingIterationsSpinBox = QtWidgets.QSpinBox()
        thermalizingIterationsSpinBox.setMinimum(1)
        thermalizingIterationsSpinBox.setMaximum(500)
        thermalizingIterationsSpinBox.setValue(100)
        thermalizingIterationsLayout.addWidget(thermalizingIterationsSpinBox)
        self.thermalizingIterationsSpinBox = thermalizingIterationsSpinBox
        coolingRateLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(coolingRateLayout)
        coolingRateLabel = QtWidgets.QLabel("Cooling rate:")
        coolingRateLayout.addWidget(coolingRateLabel)
        coolingRateSpinBox = QtWidgets.QDoubleSpinBox()
        coolingRateSpinBox.setRange(0.8, 0.99)
        coolingRateSpinBox.setSingleStep(0.01)
        coolingRateSpinBox.setDecimals(2)
        coolingRateSpinBox.setValue(0.9)
        coolingRateLayout.addWidget(coolingRateSpinBox)
        self.coolingRateSpinBox = coolingRateSpinBox
        initialRouteLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(initialRouteLayout)
        initialRouteLabel = QtWidgets.QLabel("Initial route:")
        initialRouteLayout.addWidget(initialRouteLabel)
        initialRouteSelectionBox = QtWidgets.QComboBox()
        initialRouteLayout.addWidget(initialRouteSelectionBox)
        initialRouteSelectionBox.addItem("Ordered")
        initialRouteSelectionBox.addItem("Random")
        self.initialRouteSelectionBox = initialRouteSelectionBox
        newRouteLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(newRouteLayout)
        newRouteLabel = QtWidgets.QLabel("New routes:")
        newRouteLayout.addWidget(newRouteLabel)
        newRouteSelectionBox = QtWidgets.QComboBox()
        newRouteLayout.addWidget(newRouteSelectionBox)
        newRouteSelectionBox.addItem("Random")
        newRouteSelectionBox.addItem("Swap")
        newRouteSelectionBox.addItem("Inverse")
        self.newRouteSelectionBox = newRouteSelectionBox
        stopingCriteriaSimulatedAnnealingLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(stopingCriteriaSimulatedAnnealingLayout)
        stopingCriteriaSimulatedAnnealingLabel = QtWidgets.QLabel("Stoping criteria:")
        stopingCriteriaSimulatedAnnealingLayout.addWidget(stopingCriteriaSimulatedAnnealingLabel)
        stopingCriteriaSimulatedAnnealingSpinBox = QtWidgets.QSpinBox()
        stopingCriteriaSimulatedAnnealingSpinBox.setMinimum(2)
        stopingCriteriaSimulatedAnnealingSpinBox.setMaximum(10000)
        stopingCriteriaSimulatedAnnealingSpinBox.setValue(5)
        stopingCriteriaSimulatedAnnealingLayout.addWidget(stopingCriteriaSimulatedAnnealingSpinBox)
        self.stopingCriteriaSimulatedAnnealingSpinBox = stopingCriteriaSimulatedAnnealingSpinBox
        solverConfWidgetGeneticAlgorithm = QtWidgets.QWidget()
        solverConfLayout.addWidget(solverConfWidgetGeneticAlgorithm)
        solverConfLayoutGeneticAlgorithm = QtWidgets.QVBoxLayout(solverConfWidgetGeneticAlgorithm)
        solverConfLayoutGeneticAlgorithm.setContentsMargins(0,11,11,11)
        populationSizeGeneticAlgorithmLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutGeneticAlgorithm.addLayout(populationSizeGeneticAlgorithmLayout)
        populationSizeGeneticAlgorithmLabel =QtWidgets.QLabel("Population size:")
        populationSizeGeneticAlgorithmLayout.addWidget(populationSizeGeneticAlgorithmLabel)
        populationSizeGeneticAlgorithmSpinBox = QtWidgets.QSpinBox()
        populationSizeGeneticAlgorithmSpinBox.setMinimum(10)
        populationSizeGeneticAlgorithmSpinBox.setMaximum(1000)
        populationSizeGeneticAlgorithmSpinBox.setValue(100)
        populationSizeGeneticAlgorithmLayout.addWidget(populationSizeGeneticAlgorithmSpinBox)
        self.populationSizeGeneticAlgorithmSpinBox = populationSizeGeneticAlgorithmSpinBox
        elitePercentageGeneticAlgorithmLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutGeneticAlgorithm.addLayout(elitePercentageGeneticAlgorithmLayout)
        elitePercentageGeneticAlgorithmLabel =QtWidgets.QLabel("Elite percentage:")
        elitePercentageGeneticAlgorithmLayout.addWidget(elitePercentageGeneticAlgorithmLabel)
        elitePercentageGeneticAlgorithmSpinBox = QtWidgets.QDoubleSpinBox()
        elitePercentageGeneticAlgorithmSpinBox.setRange(0,1)
        elitePercentageGeneticAlgorithmSpinBox.setSingleStep(0.01)
        elitePercentageGeneticAlgorithmSpinBox.setDecimals(2)
        elitePercentageGeneticAlgorithmSpinBox.setValue(0.05)
        elitePercentageGeneticAlgorithmLayout.addWidget(elitePercentageGeneticAlgorithmSpinBox)
        self.elitePercentageGeneticAlgorithmSpinBox = elitePercentageGeneticAlgorithmSpinBox
        mutationRateGeneticAlgorithmLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutGeneticAlgorithm.addLayout(mutationRateGeneticAlgorithmLayout)
        mutationRateGeneticAlgorithmLabel =QtWidgets.QLabel("Mutation rate:")
        mutationRateGeneticAlgorithmLayout.addWidget(mutationRateGeneticAlgorithmLabel)
        mutationRateGeneticAlgorithmSpinBox = QtWidgets.QDoubleSpinBox()
        mutationRateGeneticAlgorithmSpinBox.setRange(0,1)
        mutationRateGeneticAlgorithmSpinBox.setSingleStep(0.01)
        mutationRateGeneticAlgorithmSpinBox.setDecimals(2)
        mutationRateGeneticAlgorithmSpinBox.setValue(0.05)
        mutationRateGeneticAlgorithmLayout.addWidget(mutationRateGeneticAlgorithmSpinBox)
        self.mutationRateGeneticAlgorithmSpinBox = mutationRateGeneticAlgorithmSpinBox
        stopingCriteriaGeneticAlgorithmLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutGeneticAlgorithm.addLayout(stopingCriteriaGeneticAlgorithmLayout)
        stopingCriteriaGeneticAlgorithmLabel = QtWidgets.QLabel("Stoping criteria:")
        stopingCriteriaGeneticAlgorithmLayout.addWidget(stopingCriteriaGeneticAlgorithmLabel)
        stopingCriteriaGeneticAlgorithmSpinBox = QtWidgets.QSpinBox()
        stopingCriteriaGeneticAlgorithmSpinBox.setMinimum(10)
        stopingCriteriaGeneticAlgorithmSpinBox.setMaximum(10000)
        stopingCriteriaGeneticAlgorithmSpinBox.setValue(100)
        stopingCriteriaGeneticAlgorithmLayout.addWidget(stopingCriteriaGeneticAlgorithmSpinBox)
        self.stopingCriteriaGeneticAlgorithmSpinBox = stopingCriteriaGeneticAlgorithmSpinBox
        
        self.solverConfLayout = solverConfLayout
        solveItButton = QtWidgets.QPushButton("Solve it!")
        solveItButton.setEnabled(False)
        solverLayout.addWidget(solveItButton)
        solveItButton.clicked.connect(self.solveItButtonClicked)
        self.solveItButton = solveItButton
        
        solutionsGroup = QtWidgets.QGroupBox("Solutions:")
        solutionsGroup.setMinimumSize(QtCore.QSize(220,0))
        solutionsGroup.setMaximumSize(QtCore.QSize(220,150))
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
        solutionExecutionTimeLabel = QtWidgets.QLabel("Exec. time (s): ")
        solutionsLayout.addWidget(solutionExecutionTimeLabel)
        self.solutionSelectionBox = solutionSelectionBox
        self.solutionRouteLabel = solutionRouteLabel
        self.solutionDistanceLabel = solutionDistanceLabel
        self.solutionExecutionTimeLabel = solutionExecutionTimeLabel

        graphisGroup = QtWidgets.QGroupBox("Graphs:")
        rightLayout.addWidget(graphisGroup)
        graphicsLayout = QtWidgets.QHBoxLayout(graphisGroup)
        mapOfCitiesLayout = QtWidgets.QVBoxLayout()
        graphicsLayout.addLayout(mapOfCitiesLayout)
        mapOfCitiesCanvas = FigureCanvas(Figure(figsize=(5, 5)))
        mapOfCitiesLayout.addWidget(mapOfCitiesCanvas)
        mapOfCitiesLayout.addWidget(NavigationToolbar2QT(mapOfCitiesCanvas, self))
        convergenceLayout = QtWidgets.QVBoxLayout()
        graphicsLayout.addLayout(convergenceLayout)
        convergenceCanvas = FigureCanvas(Figure(figsize=(5, 5)))
        convergenceLayout.addWidget(convergenceCanvas)
        convergenceLayout.addWidget(NavigationToolbar2QT(convergenceCanvas, self))
        
        self.mapOfCitiesCanvas = mapOfCitiesCanvas
        self.mapOfCitiesAxes = mapOfCitiesCanvas.figure.subplots()
        self.mapOfCitiesCanvas.setMinimumSize(QtCore.QSize(500,500))
        self.mapOfCitiesAxes.set_aspect('equal')
        self.mapOfCitiesAxes.set(xlim=(0,1), ylim=(0,1))
        self.mapOfCitiesAxes.set_title('Map of cities')
        self.mapOfCitiesAxes.set_xlabel('X position (A.U.)')
        self.mapOfCitiesAxes.set_ylabel('Y position (A.U.)')
        self.mapOfCitiesScatter = self.mapOfCitiesAxes.scatter([], [], c='r')
        self.mapOfCitiesRoute, = self.mapOfCitiesAxes.plot([], [], c='grey')
        self.convergenceCanvas = convergenceCanvas
        self.convergenceAxes = convergenceCanvas.figure.subplots()
        self.convergenceCanvas.setMinimumSize(QtCore.QSize(500,500))
        #self.convergenceAxes.set_aspect('auto')
        self.convergenceAxes.set_title('Algorithm convergence')
        self.convergenceAxes.set_xlabel('Number of iterations')
        self.convergenceAxes.set_ylabel('Total route distance (A.U.)')
        self.convergenceCurve, = self.convergenceAxes.plot([], [], c='black')
        self.convergenceAxes.set_xscale('log')

        self.solutionsExactRoute = []
        self.solutionsExactDistance = []
        self.solutionsExactExecutionTime = []
        self.solutionsClosestNeighbourRoute = []
        self.solutionsClosestNeighbourDistance = []
        self.solutionsClosestNeighbourExecutionTime = []
        self.solutionsSimulatedAnnealingRoute = []
        self.solutionsSimulatedAnnealingDistance = []
        self.solutionsSimulatedAnnealingExecutionTime = []
        
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
        
        self.mapOfCitiesRoute.set_xdata(cityXpos[list(range(numberOfCities))+[0]])
        self.mapOfCitiesRoute.set_ydata(cityYpos[list(range(numberOfCities))+[0]])
        self.mapOfCitiesScatter.set_offsets(np.column_stack((cityXpos, cityYpos)))
        for child in self.mapOfCitiesAxes.get_children():
            if isinstance(child, mpl.text.Annotation):
                child.remove()
        for i in range(numberOfCities):
            self.mapOfCitiesAxes.annotate(i+1, xy=(cityXpos[i], cityYpos[i]), fontsize=15, xytext=(5,5), textcoords='offset points')
        self.mapOfCitiesScatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
        self.algorithmSelectionBox.setEnabled(True)
        if numberOfCities<=10 or self.algorithmSelectionBox.currentText() != "Exact":
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
        self.solutionsSimulatedAnnealingRoute = []
        self.solutionsSimulatedAnnealingDistance = []
        self.solutionsSimulatedAnnealingExecutionTime = []
        self.solutionSelectionBox.clear()
        self.solutionSelectionBox.setEnabled(False)
        self.solutionRouteLabel.setText("Route: ")
        self.solutionDistanceLabel.setText("Distance: ")
        self.solutionExecutionTimeLabel.setText("Exec. time (s): ")
            
        for i in range(self.initialRouteSelectionBox.count()):
            if self.initialRouteSelectionBox.itemText(i) == "Closest neighbour":
                self.initialRouteSelectionBox.removeItem(i)
            
        self.numberOfCities = numberOfCities
        self.cityXpos = cityXpos
        self.cityYpos = cityYpos
        self.distance = distance
        
    def algorithmSelectedChanged(self):
        
        if hasattr(self, 'numberOfCities'):
            if self.algorithmSelectionBox.currentText() == "Exact" and self.numberOfCities<=10 and self.solutionsExactRoute == []:
                self.solverConfLayout.setCurrentIndex(0)
                self.solveItButton.setEnabled(True)
            elif self.algorithmSelectionBox.currentText() == "Closest neighbour" and self.solutionsClosestNeighbourRoute == []:
                self.solverConfLayout.setCurrentIndex(0)
                self.solveItButton.setEnabled(True)
            elif self.algorithmSelectionBox.currentText() == "Simulated annealing":
                self.solverConfLayout.setCurrentIndex(1)
                self.solveItButton.setEnabled(True)
            elif self.algorithmSelectionBox.currentText() == "Genetic algorithm":
                self.solverConfLayout.setCurrentIndex(2)
                self.solveItButton.setEnabled(True)
            else:
                self.solveItButton.setEnabled(False)
        else:
            return
    
    def solveItButtonClicked(self):
        
        print(self.algorithmSelectionBox.currentText())
        if self.algorithmSelectionBox.currentText() == "Exact":
            self.solveExact()
        elif self.algorithmSelectionBox.currentText() == "Closest neighbour":
            self.solveNearestNeighbour()
        elif self.algorithmSelectionBox.currentText() == "Simulated annealing":
            self.solveSimulatedAnnealing()
        elif self.algorithmSelectionBox.currentText() == "Genetic algorithm":
            self.solveGeneticAlgorithm()
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
                self.mapOfCitiesRoute.set_xdata(self.cityXpos[route])
                self.mapOfCitiesRoute.set_ydata(self.cityYpos[route])
                self.mapOfCitiesScatter.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()
                
            
        print("Best route: ", bestRoute, "(", bestRouteDistance, ")")
        self.mapOfCitiesRoute.set_xdata(self.cityXpos[bestRoute])
        self.mapOfCitiesRoute.set_ydata(self.cityYpos[bestRoute])
        self.mapOfCitiesScatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
        finishTime = time.time()
        
        self.solutionsExactRoute = bestRoute
        self.solutionsExactDistance = bestRouteDistance
        self.solutionsExactExecutionTime = finishTime-startTime
        self.solutionSelectionBox.addItem("Exact")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in bestRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(bestRouteDistance))
        self.solutionExecutionTimeLabel.setText("Exec. time (s): " + str(finishTime-startTime))
        
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
                self.mapOfCitiesRoute.set_xdata(self.cityXpos[route])
                self.mapOfCitiesRoute.set_ydata(self.cityYpos[route])
                self.mapOfCitiesScatter.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()
            if routeTotalDistance<bestRouteDistance:
                bestRoute = route
                bestRouteDistance = routeTotalDistance
            
        print("Best route: ", bestRoute, "(", bestRouteDistance, ")")
        self.mapOfCitiesRoute.set_xdata(self.cityXpos[bestRoute])
        self.mapOfCitiesRoute.set_ydata(self.cityYpos[bestRoute])
        self.mapOfCitiesScatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
        finishTime = time.time() 
        
        self.solutionsClosestNeighbourRoute = bestRoute
        self.solutionsClosestNeighbourDistance = bestRouteDistance
        self.solutionsClosestNeighbourExecutionTime = finishTime-startTime
        self.solutionSelectionBox.addItem("Closest neighbour")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in bestRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(bestRouteDistance))
        self.solutionExecutionTimeLabel.setText("Exec. time (s): " + str(finishTime-startTime))
        
        self.initialRouteSelectionBox.addItem("Closest neighbour")
        
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        
    def solveSimulatedAnnealing(self):
        
        # disable elements of the GUI during the execution of the algorithm
        self.numberOfCitiesSpinBox.setEnabled(False)
        self.generateButton.setEnabled(False)
        self.algorithmSelectionBox.setEnabled(False)
        self.solveItButton.setEnabled(False)
        self.solutionSelectionBox.setEnabled(False)
        self.initialTemperatureSpinBox.setEnabled(False)
        self.thermalizingIterationsSpinBox.setEnabled(False)
        self.coolingRateSpinBox.setEnabled(False)
        self.stopingCriteriaSimulatedAnnealingSpinBox.setEnabled(False)
        self.initialRouteSelectionBox.setEnabled(False)
        self.newRouteSelectionBox.setEnabled(False)
        
        startTime = time.time()
        
        # get user defined configuration from GUI elements
        numberOfCities = self.numberOfCities
        temperature = self.initialTemperatureSpinBox.value()
        thermalizingIterations = self.thermalizingIterationsSpinBox.value()
        coolingRate = self.coolingRateSpinBox.value()
        maxThermalizationCyclesWithoutUpdate = self.stopingCriteriaSimulatedAnnealingSpinBox.value()
        initialRouteMethod = self.initialRouteSelectionBox.currentText()
        newRouteMethod = self.newRouteSelectionBox.currentText()
        
        # generate initial route according to user solutionSelectionBox
        if initialRouteMethod == "Ordered":
            # generate ordered sequence from 0 to numberOfCities-1
            oldRoute = list(range(self.numberOfCities))
            # append initial city to the route (circular route)
            oldRoute.append(oldRoute[0])
        elif initialRouteMethod == "Random":
            # generate initial route randomly
            auxArray = np.random.uniform(0,1,self.numberOfCities)
            oldRoute = list(np.argsort(auxArray))
            # append initial city to the route (circular route)
            oldRoute.append(oldRoute[0])
        elif initialRouteMethod == "Closest neighbour":
            # select route obtained in the Closest neighbour algorithm
            oldRoute = self.solutionsClosestNeighbourRoute
            
        # evaluate initial route total distance
        oldRouteTotalDistance = 0
        for idx in range(numberOfCities):
            i = oldRoute[idx]
            j = oldRoute[idx+1]
            oldRouteTotalDistance = oldRouteTotalDistance + self.distance[i][j]
        
        # initialize counter for convergence criteria
        thermalizationCyclesWithoutUpdate = 0
        iterationArray = [0]
        totalDistanceArray = [oldRouteTotalDistance]
        
        # main algorithm cycle
        while thermalizationCyclesWithoutUpdate < maxThermalizationCyclesWithoutUpdate:
            routeBeforeThermalization = copy.deepcopy(oldRoute)
            totalDistanceBeforeThermalization = oldRouteTotalDistance
            for thermalizationIndex in range(thermalizingIterations):
                # generate new route
                if newRouteMethod == "Random":
                    # random new route
                    auxArray = np.random.uniform(0,1,numberOfCities)
                    newRoute = list(np.argsort(auxArray))
                    newRoute.append(newRoute[0])
                elif newRouteMethod == "Swap":
                    # generate random indices for swap and inverse operators
                    i = np.random.choice(range(numberOfCities-1))
                    while i==0:
                        i = np.random.choice(range(numberOfCities))
                    j = np.random.choice(range(numberOfCities-1))
                    while j==i or j==0:
                        j = np.random.choice(range(numberOfCities))
                    # swap new route
                    newRoute = copy.copy(oldRoute)
                    newRoute[j], newRoute[i] = newRoute[i], newRoute[j]
                elif newRouteMethod == "Inverse":
                    # generate random indices for swap and inverse operators
                    i = np.random.choice(range(numberOfCities-1))
                    while i==0:
                        i = np.random.choice(range(numberOfCities))
                    j = np.random.choice(range(numberOfCities-1))
                    while j==i or j==0:
                        j = np.random.choice(range(numberOfCities))
                    # inverse new route
                    newRoute = copy.copy(oldRoute)
                    newRoute[min(i,j):max(i,j)+1] = newRoute[max(i,j):min(i,j)-1:-1]
                
                # evaluate new route total distance
                newRouteTotalDistance = 0
                for idx in range(numberOfCities):
                    initialCity = newRoute[idx]
                    finalCity = newRoute[idx+1]
                    newRouteTotalDistance = newRouteTotalDistance + self.distance[initialCity][finalCity]
                    
                # decide wheter the new route gets selected or not
                if newRouteTotalDistance <= oldRouteTotalDistance:
                    oldRoute = copy.copy(newRoute)
                    oldRouteTotalDistance = newRouteTotalDistance
                    if self.liveSolverCheckBox.isChecked():
                        self.mapOfCitiesRoute.set_xdata(self.cityXpos[oldRoute])
                        self.mapOfCitiesRoute.set_ydata(self.cityYpos[oldRoute])
                        self.mapOfCitiesScatter.figure.canvas.draw()
                        QtCore.QCoreApplication.processEvents()
                elif np.random.uniform(0,1,1)<np.exp(-(newRouteTotalDistance-oldRouteTotalDistance)/temperature):
                    oldRoute = copy.copy(newRoute)
                    oldRouteTotalDistance = newRouteTotalDistance
                    if self.liveSolverCheckBox.isChecked():
                        self.mapOfCitiesRoute.set_xdata(self.cityXpos[oldRoute])
                        self.mapOfCitiesRoute.set_ydata(self.cityYpos[oldRoute])
                        self.mapOfCitiesScatter.figure.canvas.draw()
                        QtCore.QCoreApplication.processEvents()

                # update convergence data and graph
                iterationArray.append(iterationArray[-1]+1)
                totalDistanceArray.append(oldRouteTotalDistance)
                if self.liveSolverCheckBox.isChecked():
                    self.convergenceCurve.set_xdata(iterationArray)
                    self.convergenceCurve.set_ydata(totalDistanceArray)
                    self.convergenceAxes.set(xlim=(1,iterationArray[-1]), ylim=(0,np.amax(totalDistanceArray)))
                    self.convergenceAxes.figure.canvas.draw()
                    QtCore.QCoreApplication.processEvents()
            
            if oldRoute != routeBeforeThermalization:
                thermalizationCyclesWithoutUpdate = 0
            else:
                thermalizationCyclesWithoutUpdate = thermalizationCyclesWithoutUpdate + 1
                
            temperature = temperature*coolingRate
            print(thermalizationCyclesWithoutUpdate, temperature)
            
        self.mapOfCitiesRoute.set_xdata(self.cityXpos[oldRoute])
        self.mapOfCitiesRoute.set_ydata(self.cityYpos[oldRoute])
        self.mapOfCitiesScatter.figure.canvas.draw()
        self.convergenceCurve.set_xdata(iterationArray)
        self.convergenceCurve.set_ydata(totalDistanceArray)
        self.convergenceAxes.set(xlim=(1,iterationArray[-1]), ylim=(0,np.amax(totalDistanceArray)))
        self.convergenceAxes.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        print("Best route: ", oldRoute, "(", oldRouteTotalDistance, ")")

        finishTime = time.time()
        
        self.solutionsSimulatedAnnealingRoute = oldRoute
        self.solutionsSimulatedAnnealingDistance = oldRouteTotalDistance
        self.solutionsSimulatedAnnealingExecutionTime = finishTime-startTime
        for i in range(self.solutionSelectionBox.count()):
            if self.solutionSelectionBox.itemText(i) == "Simulated annealing":
                self.solutionSelectionBox.removeItem(i)
        self.solutionSelectionBox.addItem("Simulated annealing")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in oldRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(oldRouteTotalDistance))
        self.solutionExecutionTimeLabel.setText("Exec. time (s): " + str(finishTime-startTime))
        
        # enable elements of the GUI after the execution of the GA algorithm
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solveItButton.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        self.initialTemperatureSpinBox.setEnabled(True)
        self.thermalizingIterationsSpinBox.setEnabled(True)
        self.coolingRateSpinBox.setEnabled(True)
        self.stopingCriteriaSimulatedAnnealingSpinBox.setEnabled(True)
        self.initialRouteSelectionBox.setEnabled(True)
        self.newRouteSelectionBox.setEnabled(True)
        
        
    def solveGeneticAlgorithm(self):
        
        # disable elements of the GUI during the execution of the algorithm
        self.numberOfCitiesSpinBox.setEnabled(False)
        self.generateButton.setEnabled(False)
        self.algorithmSelectionBox.setEnabled(False)
        self.solveItButton.setEnabled(False)
        self.solutionSelectionBox.setEnabled(False)
        self.populationSizeGeneticAlgorithmSpinBox.setEnabled(False)
        self.elitePercentageGeneticAlgorithmSpinBox.setEnabled(False)
        self.mutationRateGeneticAlgorithmSpinBox.setEnabled(False)
        self.stopingCriteriaGeneticAlgorithmSpinBox.setEnabled(False)
        
        startTime = time.time()
        
        # get user defined configuration from GUI elements
        numberOfCities = self.numberOfCities
        populationSize = self.populationSizeGeneticAlgorithmSpinBox.value()
        mutationRate = self.mutationRateGeneticAlgorithmSpinBox.value()
        elitePercentage = self.elitePercentageGeneticAlgorithmSpinBox.value()
        eliteSize = max([1, int(round(populationSize*elitePercentage))])
        maxIterationsWithoutUpdate = self.stopingCriteriaGeneticAlgorithmSpinBox.value()
        
        # INITIALIZATION: create initial population for the GA
        # generate initial population (route, total route distance, fitness and cumulative probability)
        oldGeneration = [[]]
        oldGenerationDistances = []
        fitness = []
        cumulativeProbabilities = []
        norm = 0
        for i in range(populationSize):
            # generate routes in initial population randomly
            auxArray = np.random.uniform(0,1,numberOfCities)
            route = list(np.argsort(auxArray))
            # append initial city to the route (circular route)
            route.append(route[0])
            # evaluate route distance, fitness and probability (cumulative) of breeding
            totalDistance = 0
            for i in range(numberOfCities):
                totalDistance = totalDistance + self.distance[route[i]][route[i+1]]
            # evaluate cummulative probability (not normalized) by adding fitness to the norm
            norm = norm + 1./totalDistance
            # append route, total distance and cumulative probability to the population arrays
            oldGeneration.append(route)
            oldGenerationDistances.append(totalDistance)
            fitness.append(1./totalDistance)
            cumulativeProbabilities.append(norm)
        # remove first route (empty) from old generation
        oldGeneration.remove([])
        # renormalize cumulative probability to 1
        for i in range(populationSize):
            cumulativeProbabilities[i] = cumulativeProbabilities[i]/norm
        # rank old generation routes
        oldGenerationRank = np.argsort(oldGenerationDistances)
        
        # select best route in the population
        bestRoute = copy.deepcopy(oldGeneration[oldGenerationRank[0]])
        bestRouteDistance = oldGenerationDistances[oldGenerationRank[0]]
        
        # initialize counter for convergence criteria
        iterationsWithoutUpdate = 0
        iterationArray = [0]
        totalDistanceArray = [oldGenerationDistances[oldGenerationRank[0]]]
        
        # main algorithm cycle
        while iterationsWithoutUpdate < maxIterationsWithoutUpdate:
            # create new generation
            newGeneration = [[]]
            newGenerationDistances = []
            
            # ELITISM: copy elite population (most fit) over to the next generation
            for i in range(eliteSize):
                newGeneration.append(copy.deepcopy(oldGeneration[oldGenerationRank[i]]))
            
            # BREEDING: breed individuals of the old generation to complete the new generation 
            for i in range(eliteSize, populationSize):
                # select mating partners according to their probability of reproduction (proportional to their fitness)
                rnd = np.random.uniform()
                p1 = 0
                for i in range(1, populationSize):
                    if rnd>cumulativeProbabilities[i-1] and rnd<=cumulativeProbabilities[i]:
                        p1 = i
                        break
                while True:
                    rnd = np.random.uniform()
                    p2 = 0
                    for i in range(1, populationSize):
                        if rnd>cumulativeProbabilities[i-1] and rnd<=cumulativeProbabilities[i]:
                            p2 = i
                            break
                    if p2 != p1:
                        break
                # create child from matting partners by crossover of their genes
                # select genes to copy from p1
                genMin = np.random.choice(self.numberOfCities)
                genMax = np.random.choice(self.numberOfCities)
                while genMax == genMin:
                    genMax = np.random.choice(self.numberOfCities)
                genMin, genMax = min(genMin, genMax), max(genMin, genMax)
                genesP1 = oldGeneration[p1][genMin:genMax+1]
                # select genes to copy from p2
                genesP2 = [gen for gen in oldGeneration[p2] if gen not in genesP1]
                # create offspring
                offspring = []
                for gen in range(self.numberOfCities):
                    if gen < genMin:
                        offspring.append(genesP2[gen])
                    elif gen <= genMax:
                        offspring.append(genesP1[gen-genMin])
                    else:
                        offspring.append(genesP2[gen-(genMax-genMin)-1])
                offspring.append(offspring[0])
                # append offspring to new generation
                newGeneration.append(copy.deepcopy(offspring))
        
            # remove first (empty) route in newGeneration
            newGeneration.remove([])
            
            # MUTATION: mutate individuals of the new generation according to mutation rate
            for i in range(eliteSize, populationSize):
                if np.random.uniform() <= mutationRate:
                    firstCity = np.random.choice(range(self.numberOfCities))
                    secondCity = np.random.choice(range(self.numberOfCities))
                    while secondCity == firstCity:
                        secondCity = np.random.choice(range(self.numberOfCities))
                    newGeneration[i][firstCity], newGeneration[i][secondCity] = newGeneration[i][secondCity], newGeneration[i][firstCity]
                    newGeneration[i][-1] = newGeneration[i][0]
            
            # evaluate total distance of each chromosome in the population
            for i in range(populationSize):
                totalDistance = 0
                for j in range(self.numberOfCities):
                    totalDistance = totalDistance + self.distance[newGeneration[i][j]][newGeneration[i][j+1]]
                newGenerationDistances.append(totalDistance)
            
            # RANKING: evaluate fitness of the new generation and probability (cumulative) of breeding
            newGenerationRank = np.argsort(newGenerationDistances)
            fitness = []
            cumulativeProbabilities = []
            norm = 0
            for i in range(populationSize):
                fitness.append(1./newGenerationDistances[i])
                cumulativeProbabilities.append(norm+fitness[i])
                norm = norm + fitness[i]
            for i in range(populationSize):
                cumulativeProbabilities[i] = cumulativeProbabilities[i]/norm
            
            # select fittest individual
            if newGenerationDistances[newGenerationRank[0]] < bestRouteDistance:
                bestRoute = copy.deepcopy(newGeneration[newGenerationRank[0]])
                bestRouteDistance = newGenerationDistances[newGenerationRank[0]]
                iterationsWithoutUpdate = 0
            elif newGenerationDistances[newGenerationRank[0]] == bestRouteDistance and newGeneration[newGenerationRank[0]] != bestRoute:
                bestRoute = copy.deepcopy(newGeneration[newGenerationRank[0]])
                bestRouteDistance = newGenerationDistances[newGenerationRank[0]]
                iterationsWithoutUpdate = 0
            else:
                iterationsWithoutUpdate = iterationsWithoutUpdate+1
                
            # copy new generation into old generation
            for i in range(populationSize):
                oldGeneration[i] = copy.deepcopy(newGeneration[i])
                oldGenerationDistances[i] = newGenerationDistances[i]
                oldGenerationRank[i] = newGenerationRank[i]
            
            # output
            iterationArray.append(iterationArray[-1]+1)
            totalDistanceArray.append(newGenerationDistances[newGenerationRank[0]])
            print("Fittest chromosome of ", iterationArray[-1], " generation ->", bestRoute, "(", bestRouteDistance, ")")
            if self.liveSolverCheckBox.isChecked():
                self.mapOfCitiesRoute.set_xdata(self.cityXpos[newGeneration[newGenerationRank[0]]])
                self.mapOfCitiesRoute.set_ydata(self.cityYpos[newGeneration[newGenerationRank[0]]])
                self.mapOfCitiesScatter.figure.canvas.draw()
                self.convergenceCurve.set_xdata(iterationArray)
                self.convergenceCurve.set_ydata(totalDistanceArray)
                self.convergenceAxes.set(xlim=(1,iterationArray[-1]), ylim=(0,np.amax(totalDistanceArray)))
                self.convergenceAxes.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()

        finishTime = time.time()
        
        self.solutionsGeneticAlgorithmRoute = bestRoute
        self.solutionsGeneticAlgorithmDistance = bestRouteDistance
        self.solutionsGeneticAlgorithmExecutionTime = finishTime-startTime
        for i in range(self.solutionSelectionBox.count()):
            if self.solutionSelectionBox.itemText(i) == "Genetic algorithm":
                self.solutionSelectionBox.removeItem(i)
        self.solutionSelectionBox.addItem("Genetic algorithm")
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in bestRoute]))
        self.solutionDistanceLabel.setText("Distance: " + str(bestRouteDistance))
        self.solutionExecutionTimeLabel.setText("Exec. time (s): " + str(finishTime-startTime))
        
        # enable elements of the GUI after the execution of the GA algorithm
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solveItButton.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        self.populationSizeGeneticAlgorithmSpinBox.setEnabled(True)
        self.elitePercentageGeneticAlgorithmSpinBox.setEnabled(True)
        self.mutationRateGeneticAlgorithmSpinBox.setEnabled(True)
        self.stopingCriteriaGeneticAlgorithmSpinBox.setEnabled(True)

        
    def solutionSelectedChanged(self):
        
        if self.solutionSelectionBox.currentText() == "Exact":
            route = self.solutionsExactRoute
            distance = self.solutionsExactDistance
            executionTime = self.solutionsExactExecutionTime
        elif self.solutionSelectionBox.currentText() == "Closest neighbour":
            route = self.solutionsClosestNeighbourRoute
            distance = self.solutionsClosestNeighbourDistance
            executionTime = self.solutionsClosestNeighbourExecutionTime
        elif self.solutionSelectionBox.currentText() == "Simulated annealing":
            route = self.solutionsSimulatedAnnealingRoute
            distance = self.solutionsSimulatedAnnealingDistance
            executionTime = self.solutionsSimulatedAnnealingExecutionTime
        elif self.solutionSelectionBox.currentText() == "Genetic algorithm":
            route = self.solutionsGeneticAlgorithmRoute
            distance = self.solutionsGeneticAlgorithmDistance
            executionTime = self.solutionsGeneticAlgorithmExecutionTime
        self.solutionRouteLabel.setText("Route: " + str([i+1 for i in route]))
        self.solutionDistanceLabel.setText("Distance: " + str(distance))
        self.solutionExecutionTimeLabel.setText("Exec. Time (s): " + str(executionTime))
        self.mapOfCitiesRoute.set_xdata(self.cityXpos[route])
        self.mapOfCitiesRoute.set_ydata(self.cityYpos[route])
        self.mapOfCitiesScatter.figure.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        
            
qapp = QtWidgets.QApplication(sys.argv)
appWindow = MainWindow()
appWindow.show()
appWindow.activateWindow()
appWindow.raise_()
qapp.exec()
