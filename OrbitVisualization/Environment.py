import matplotlib.pyplot as plt
import numpy as np

class Planet:
    def __init__(self, radius, env_width, grid=False, wireframe=False, darkmode=True):

        # Object variables
        self.radius = radius
        self.env_width = env_width
        
        # Colors
        linecolor = 'black'
        dark = '#181818'
        planet = '#2B4EFF'

        # Create figure and set parameters
        self.fig = plt.figure()
        ax = self.fig.add_subplot(projection='3d')

        ax.set_box_aspect([1,1,0.9])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Grid handling
        if not grid:
            plt.axis('off')

        # Darkmode handling
        if darkmode:   
            linecolor = 'white'
            planet = '#162780'

            self.fig.patch.set_facecolor(dark)
            ax.set_facecolor(dark)

            if grid:
                ax.w_xaxis.line.set_color('white')
                ax.w_yaxis.line.set_color('white')
                ax.w_zaxis.line.set_color('white')

                ax.tick_params(colors='white')
            
        # Generate sphere graphing data
        theta, phi = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]

        x = radius*np.cos(theta)*np.sin(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(phi)

        # Wireframe handling
        if wireframe:
            ax.plot_wireframe(x, y, z, color=linecolor, linewidth=0.5)
        else:
            ax.plot_surface(x, y, z, color=planet, edgecolor=linecolor, linewidth=0.5)