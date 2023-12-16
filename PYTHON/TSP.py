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
        algorithmSelectionBox.addItem("Nearest neighbour")
        algorithmSelectionBox.addItem("Simulated annealing")
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
        initialTemperatureSpinBox.setRange(0.1,1000)
        initialTemperatureSpinBox.setSingleStep(0.1)
        initialTemperatureSpinBox.setDecimals(1)
        initialTemperatureLayout.addWidget(initialTemperatureSpinBox)
        self.initialTemperatureSpinBox = initialTemperatureSpinBox
        coolingRateLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(coolingRateLayout)
        coolingRateLabel = QtWidgets.QLabel("Cooling rate:")
        coolingRateLayout.addWidget(coolingRateLabel)
        coolingRateSpinBox = QtWidgets.QDoubleSpinBox()
        coolingRateSpinBox.setRange(0.8, 0.999)
        coolingRateSpinBox.setSingleStep(0.001)
        coolingRateSpinBox.setDecimals(3)
        coolingRateLayout.addWidget(coolingRateSpinBox)
        self.coolingRateSpinBox = coolingRateSpinBox
        stopingCriteriaSimulatedAnnealingLayout = QtWidgets.QHBoxLayout()
        solverConfLayoutSimulatedAnnealing.addLayout(stopingCriteriaSimulatedAnnealingLayout)
        stopingCriteriaSimulatedAnnealingLabel = QtWidgets.QLabel("Stoping criteria:")
        stopingCriteriaSimulatedAnnealingLayout.addWidget(stopingCriteriaSimulatedAnnealingLabel)
        stopingCriteriaSimulatedAnnealingSpinBox = QtWidgets.QSpinBox()
        stopingCriteriaSimulatedAnnealingSpinBox.setMinimum(10)
        stopingCriteriaSimulatedAnnealingSpinBox.setMaximum(10000)
        stopingCriteriaSimulatedAnnealingLayout.addWidget(stopingCriteriaSimulatedAnnealingSpinBox)
        self.stopingCriteriaSimulatedAnnealingSpinBox = stopingCriteriaSimulatedAnnealingSpinBox
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
        
        self.solverConfLayout = solverConfLayout
        solverConfLayoutSimulatedAnnealing.addStretch()
        solveItButton = QtWidgets.QPushButton("Solve it!")
        solveItButton.setEnabled(False)
        solverLayout.addWidget(solveItButton)
        solveItButton.clicked.connect(self.solveItButtonClicked)
        self.solveItButton = solveItButton
        
        solutionsGroup = QtWidgets.QGroupBox("Solutions:")
        solutionsGroup.setMinimumSize(QtCore.QSize(220,0))
        solutionsGroup.setMaximumSize(QtCore.QSize(220,2000))
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
            elif self.algorithmSelectionBox.currentText() == "Nearest neighbour" and self.solutionsClosestNeighbourRoute == []:
                self.solverConfLayout.setCurrentIndex(0)
                self.solveItButton.setEnabled(True)
            elif self.algorithmSelectionBox.currentText() == "Simulated annealing":
                self.solverConfLayout.setCurrentIndex(1)
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
        elif self.algorithmSelectionBox.currentText() == "Simulated annealing":
            self.solveSimulatedAnnealing()
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
        
        self.numberOfCitiesSpinBox.setEnabled(False)
        self.generateButton.setEnabled(False)
        self.algorithmSelectionBox.setEnabled(False)
        self.solveItButton.setEnabled(False)
        self.solutionSelectionBox.setEnabled(False)
        self.initialTemperatureSpinBox.setEnabled(False)
        self.coolingRateSpinBox.setEnabled(False)
        self.stopingCriteriaSimulatedAnnealingSpinBox.setEnabled(False)
        self.initialRouteSelectionBox.setEnabled(False)
        self.newRouteSelectionBox.setEnabled(False)
        
        startTime = time.time()
        
        temperature = self.initialTemperatureSpinBox.value()
        coolingRate = self.coolingRateSpinBox.value()
        
        # generate initial route according to user solutionSelectionBox
        if self.initialRouteSelectionBox.currentText() == "Ordered":
            # generate ordered sequence from 0 to numberOfCities-1
            oldRoute = list(range(self.numberOfCities))
            # append initial city to the route (circular route)
            oldRoute.append(oldRoute[0])
        elif self.initialRouteSelectionBox.currentText() == "Random":
            # generate initial route randomly
            auxArray = np.random.uniform(0,1,self.numberOfCities)
            oldRoute = list(np.argsort(auxArray))
            # append initial city to the route (circular route)
            oldRoute.append(oldRoute[0])
        elif self.initialRouteSelectionBox.currentText() == "Closest neighbour":
            # select route obtained in the Closest neighbour algorithm
            oldRoute = self.solutionsClosestNeighbourRoute
            
        # evaluate initial route total distance
        oldRouteTotalDistance = 0
        for idx in range(self.numberOfCities):
            i = oldRoute[idx]
            j = oldRoute[idx+1]
            oldRouteTotalDistance = oldRouteTotalDistance + self.distance[i][j]
        
        # initialize counter for convergence criteria
        iterationsWithoutUpdate = 0
        maxIterationsWithoutUpdate = self.stopingCriteriaSimulatedAnnealingSpinBox.value()
        
        # main algorithm cycle
        iterationArray = [0]
        totalDistanceArray = [oldRouteTotalDistance]
        while iterationsWithoutUpdate < maxIterationsWithoutUpdate:
            # generate new route
            if self.newRouteSelectionBox.currentText() == "Random":
                # random new route
                auxArray = np.random.uniform(0,1,self.numberOfCities)
                newRoute = list(np.argsort(auxArray))
                newRoute.append(newRoute[0])
            elif self.newRouteSelectionBox.currentText() == "Swap":
                # generate random indices for swap and inverse operators
                i = np.random.choice(range(self.numberOfCities-1))
                while i==0:
                    i = np.random.choice(range(self.numberOfCities))
                j = np.random.choice(range(self.numberOfCities-1))
                while j==i or j==0:
                    j = np.random.choice(range(self.numberOfCities))
                # swap new route
                newRoute = copy.copy(oldRoute)
                newRoute[j], newRoute[i] = newRoute[i], newRoute[j]
            elif self.newRouteSelectionBox.currentText() == "Inverse":
                # generate random indices for swap and inverse operators
                i = np.random.choice(range(self.numberOfCities-1))
                while i==0:
                    i = np.random.choice(range(self.numberOfCities))
                j = np.random.choice(range(self.numberOfCities-1))
                while j==i or j==0:
                    j = np.random.choice(range(self.numberOfCities))
                # inverse new route
                newRoute = copy.copy(oldRoute)
                newRoute[min(i,j):max(i,j)+1] = newRoute[max(i,j):min(i,j)-1:-1]
            
            # evaluate new route total distance
            newRouteTotalDistance = 0
            for idx in range(self.numberOfCities):
                initialCity = newRoute[idx]
                finalCity = newRoute[idx+1]
                newRouteTotalDistance = newRouteTotalDistance + self.distance[initialCity][finalCity]
                
            # decide wheter the new route gets selected or not
            if newRouteTotalDistance <= oldRouteTotalDistance:
                oldRoute = copy.copy(newRoute)
                oldRouteTotalDistance = newRouteTotalDistance
                temperature = temperature*coolingRate
                if self.liveSolverCheckBox.isChecked():
                    self.mapOfCitiesRoute.set_xdata(self.cityXpos[oldRoute])
                    self.mapOfCitiesRoute.set_ydata(self.cityYpos[oldRoute])
                    self.mapOfCitiesScatter.figure.canvas.draw()
                    QtCore.QCoreApplication.processEvents()
                iterationsWithoutUpdate = 0
            elif np.random.uniform(0,1,1)<np.exp(-(newRouteTotalDistance-oldRouteTotalDistance)/temperature):
                oldRoute = copy.copy(newRoute)
                oldRouteTotalDistance = newRouteTotalDistance
                temperature = temperature*coolingRate
                if self.liveSolverCheckBox.isChecked():
                    self.mapOfCitiesRoute.set_xdata(self.cityXpos[oldRoute])
                    self.mapOfCitiesRoute.set_ydata(self.cityYpos[oldRoute])
                    self.mapOfCitiesScatter.figure.canvas.draw()
                    QtCore.QCoreApplication.processEvents()
                iterationsWithoutUpdate = 0
            else:
                temperature = temperature*coolingRate
                iterationsWithoutUpdate = iterationsWithoutUpdate+1

            # update convergence data and graph
            iterationArray.append(iterationArray[-1]+1)
            totalDistanceArray.append(oldRouteTotalDistance)
            if self.liveSolverCheckBox.isChecked():
                self.convergenceCurve.set_xdata(iterationArray)
                self.convergenceCurve.set_ydata(totalDistanceArray)
                self.convergenceAxes.set(xlim=(1,iterationArray[-1]), ylim=(0,np.amax(totalDistanceArray)))
                self.convergenceAxes.figure.canvas.draw()
                QtCore.QCoreApplication.processEvents()
                
            print(iterationsWithoutUpdate, temperature)
            
        
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
        
        self.numberOfCitiesSpinBox.setEnabled(True)
        self.generateButton.setEnabled(True)
        self.algorithmSelectionBox.setEnabled(True)
        self.solveItButton.setEnabled(True)
        self.solutionSelectionBox.setEnabled(True)
        self.initialTemperatureSpinBox.setEnabled(True)
        self.coolingRateSpinBox.setEnabled(True)
        self.stopingCriteriaSimulatedAnnealingSpinBox.setEnabled(True)
        self.initialRouteSelectionBox.setEnabled(True)
        self.newRouteSelectionBox.setEnabled(True)
        
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
