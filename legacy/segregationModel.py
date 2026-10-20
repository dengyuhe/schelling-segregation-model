import random
import numpy as np

# This class represents one cell of a model grid
class Cell:
    # Type: Single character (ex. "#", "O")
    # Grid: The Grid that this Cell is part of
    # X,Y: Coordinates of this Cell in the Grid
    # Min: Minimum tolerated ratio of neighbors that are of same type
    # Max: Maximum tolerated ratio of neighbors that are of same type (defaults to 1.0)
    def __init__(self,symbol,min,max=1.0):
        self.symbol = symbol
        self.grid = None
        self.x = None
        self.y = None
        self.min = min
        self.max = max

    # Returns the Cell in the Grid that is offset from this cell
    # by dx,dy
    def getRelative(self,dx,dy):
        return self.grid.getCell(self.x+dx,self.y+dy)

    # Gets the ratio of neighbors who are of the same type
    def neighborhoodRatio(self,neighborhood):
        neighbors = [self.getRelative(dx,dy) for (dx,dy) in neighborhood]
        same = 0
        total = 0
        for n in neighbors:
            if n is not None:
                total += 1
                if n.symbol == self.symbol:
                    same += 1
        #print(str(self.x) + "," + str(self.y) + ": " + str(same) + "/" + str(total) + " " + str([n.symbol if n is not None else "-" for n in neighbors]))
        if (total == 0):
            return 1.0
        else:
            return same / total

    # Returns the outright fraction (IE X/Y) of neighbors who are of the same color
    # (only for use in visualizers/debugging)
    def neighborhoodFrac(self,neighborhood):
        neighbors = [self.getRelative(dx,dy) for (dx,dy) in neighborhood]
        same = 0
        total = 0
        for n in neighbors:
            if n is not None:
                total += 1
                if n.symbol == self.symbol:
                    same += 1
        #print(str(self.x) + "," + str(self.y) + ": " + str(same) + "/" + str(total) + " " + str([n.symbol if n is not None else "-" for n in neighbors]))
        return str(same) + "/" + str(total)

    def isTolerant(self,neighborhood):
        nr = self.neighborhoodRatio(neighborhood)
        return self.min <= nr <= self.max

# This class represents a whole model grid
class Grid:
    # DimX,DimY: Dimensions of grid
    # Cells: Array of cells (initialized as all empty)
    # Empties: Array of tuples corresponding to empty spaces (this is a cache for programmatic efficiency)
    def __init__(self,dimX,dimY):
        self.dimX = dimX
        self.dimY = dimY
        self.cells = [[None for y in range(dimY)] for x in range(dimX)]
        self.empties = [(x,y) for x in range(dimX) for y in range(dimY)]

    def addCell(self,x,y,cell):
        if (cell is not None):
            cell.grid = self
            cell.x = x
            cell.y = y
            self.cells[x][y] = cell
            self.empties.remove((x, y))
        else:
            self.cells[x][y] = cell
            self.empties.remove((x, y))
            self.empties.append((x, y))

    # Simple getter function to return the cell at x,y
    def getCell(self,x,y):
        if (0 <= x < self.dimX) and (0 <= y < self.dimY):
            return self.cells[x][y]
        else:
            return None

    # Move the cell located at Xi,Yi to Xf,Yf
    # Throw an error if Xf,Yf is not empty
    def moveTo(self,xi,yi,xf,yf):
        if not (xf,yf) in self.empties:
            raise ValueError("Cannot move a cell to non-empty coordinates "+str(xf)+", "+str(yf)+"!")
        c = self.getCell(xi,yi)
        self.cells[xf][yf] = c
        self.cells[xi][yi] = None
        self.empties.remove((xf,yf))
        self.empties.append((xi,yi))
        c.x = xf
        c.y = yf

    # Move the cell located at Xi,Yi to a random location on the grid
    # Location must be empty
    def move(self,xi,yi):
        i = random.randint(0,len(self.empties)-1)
        (xf,yf) = self.empties[i]
        self.moveTo(xi,yi,xf,yf)

    # Return a visual representation of this grid
    def visualize(self):
        toReturn = ""
        for y in range(self.dimY):
            for x in range(self.dimX):
                s = "-"
                c = self.getCell(x,y)
                if c is not None:
                    s = c.symbol
                toReturn = toReturn + s
            toReturn = toReturn + "\n"
        return toReturn

    # Return a visual representation of this grid
    def visualizeScores(self,neighborhood):
        toReturn = ""
        for y in range(self.dimY):
            for x in range(self.dimX):
                s = "---"
                c = self.getCell(x,y)
                if c is not None:
                    s = c.neighborhoodFrac(neighborhood)
                toReturn = toReturn + s + " "
            toReturn = toReturn + "\n"
        return toReturn

    # Return a visual representation of this grid
    def visualizeCoords(self):
        toReturn = ""
        for y in range(self.dimY):
            for x in range(self.dimX):
                s = "---"
                c = self.getCell(x,y)
                if c is not None:
                    s = str(c.x)+","+str(c.y)
                toReturn = toReturn + s + " "
            toReturn = toReturn + "\n"
        return toReturn

    # Measures the average neighborhood sameness of each nonempty cell in the grid
    def samenessIndex(self, neighborhood):
        sum = 0
        n = 0
        for y in range(self.dimY):
            for x in range(self.dimX):
                c = self.getCell(x, y)
                if c is not None:
                    sum += c.neighborhoodRatio(neighborhood)
                    n += 1
        if n != 0:
            return sum / n
        return None

# This class represents a segregation model which can be run
class Model:
    # DimX,DimY: Dimensions of grid
    # Symbols: Array of symbols (ex. ["#","O","&"]) to be used
    # Densities: Array of portions of the grid that will be taken up by each
    #    (Each must be in [0,1], and the sum must not exceed 1)
    # ToleranceFuncs: Array of functions which have return type (min, max)
    #    and must take in as their argument the quantile of the cell whose
    #    tolerances are to be established (ex. 0.8 for the cell that is 80%
    #    of the way through the list)
    # Neighborhood: List of tuples (dx,dy) determining how each Cell will define
    #    its neighborhood
    def __init__(self,dimX,dimY,symbols,densities,toleranceFuncs,neighborhood):
        # Throw errors if arguments are bad
        if not len(symbols) == len(densities) == len(toleranceFuncs):
            raise ValueError("Number of symbols (" + len(symbols) + "), densities (" + len(densities)
                             + "), and tolerance funcs (" + len(toleranceFuncs) + ") do not match!")
        if sum(densities) > 1:
            raise ValueError("Sum of densities " + str(densities) + " exceeds 1.0!")
        if False in [0 <= d <= 1 for d in densities]:
            raise ValueError("List of densities " + str(densities) + " includes a value which is outside [0,1]!")

        # Generate cells
        n = dimX * dimY
        counts = [int(c) for c in (np.floor(np.multiply(densities,n)))]
        cellList = []
        for i in range(len(counts)):
            for j in range(counts[i]):
                (cMin, cMax) = toleranceFuncs[i](j/(counts[i]-1))
                c = Cell(symbols[i],cMin,cMax)
                cellList.append(c)
        for j in range(n-sum(counts)):
            cellList.append(None)

        # Set cell locations
        self.grid = Grid(dimX, dimY)
        for x in range(dimX):
            for y in range(dimY):
                i = random.randint(0,len(cellList)-1)
                self.grid.addCell(x,y,cellList[i])
                cellList.pop(i)
        self.isRun = False
        self.neighborhood = neighborhood

    # Run the model
    def run(self,maxIterations,visualize=False):
        if (self.isRun):
            raise ValueError("Cannot run a model that has already been run!")
        self.isRun = True
        initialSameness = self.grid.samenessIndex(self.neighborhood)

        # List of cells that are currently not tolerant
        # (Another cache)
        intolerants = []
        # Initialize intolerants cache
        for x in range(self.grid.dimX):
            for y in range(self.grid.dimX):
                c = self.grid.getCell(x,y)
                if c is not None and not c.isTolerant(self.neighborhood):
                    intolerants.append((x,y))

        # Run model
        iterations = 0
        complete = False
        for i in range(maxIterations):
            iterations += 1
            # Simulation over?
            if (len(intolerants) == 0):
                complete = True
                break
            # Move a random intolerant cell
            else:
                # Select a random intolerant cell
                i = random.randint(0,len(intolerants)-1)
                (xi,yi) = intolerants[i]
                c = self.grid.getCell(xi,yi)
                # Move the random intolerant cell
                self.grid.move(xi,yi)
                (xf,yf) = c.x, c.y
                # Update intolerants cache
                intolerants.pop(i)
                for x in range(xf-1,xf+2):
                    for  y in range(yf-1,yf+2):
                        if (x,y) in intolerants:
                            intolerants.remove((x, y))
                        c = self.grid.getCell(x,y)
                        if c is not None and not c.isTolerant(self.neighborhood):
                            intolerants.append((x,y))

        # Report
        finalSameness = self.grid.samenessIndex(self.neighborhood)
        if (visualize):
            print(self.grid.visualize())
            #print(self.grid.visualizeScores(self.neighborhood))
            #print(self.grid.visualizeCoords())
        return (iterations,complete,initialSameness,finalSameness)
