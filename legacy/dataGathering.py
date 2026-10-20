import numpy as np
import matplotlib.pyplot as plt
import segregationModel as sm

neighborhood = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),(2,0),(0,2),(-2,0),(0,-2)]
#neighborhood = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
x = np.arange(0.0,0.5,0.0005)
y = []
for xi in x:
    def f(quantile):
        return (xi, 1.0)
    m = sm.Model(50, 10, ['#', 'O'], [0.45, 0.45], [f, f], neighborhood)
    (iterations, complete, initialSameness, finalSameness) = m.run(10000)
    y.append(finalSameness)
plt.plot(x,y,"b.",markersize=2)
plt.show()