from skyfield.api import EarthSatellite
from skyfield.api import load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import timedelta
import numpy as np

class Environment:
    def __init__(self, radius, env_width, start_time, duration=None, grid=False, darkmode=True):

        # Object variables
        self.satellites = []
        self.ts = load.timescale()
        self.start_time = start_time
        self.duration = duration
        if duration: 
            self.end_time = self.ts.utc(start_time.utc_datetime() + timedelta(hours=duration)) 
        else: 
            self.end_time = None
        
        # Colors
        linecolor = 'black'
        dark = '#181818'

        # Create figure and set parameters
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        plt.locator_params(nbins=3)

        self.ax.set_box_aspect([1,1,0.9])
        self.ax.view_init(10, 45)

        self.ax.set_xlim(-env_width, env_width)
        self.ax.set_ylim(-env_width, env_width)
        self.ax.set_zlim(-env_width, env_width)

        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Grid handling
        if not grid:
            plt.axis('off')

        # Darkmode handling
        if darkmode:   
            linecolor = 'white'

            self.fig.patch.set_facecolor(dark)
            self.ax.set_facecolor(dark)

            if grid:
                self.ax.w_xaxis.line.set_color('white')
                self.ax.w_yaxis.line.set_color('white')
                self.ax.w_zaxis.line.set_color('white')

                self.ax.tick_params(colors='white')
            
        # Generate and graph sphere
        theta, phi = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]

        x = radius*np.cos(theta)*np.sin(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(phi)

        self.ax.plot_wireframe(x, y, z, color=linecolor, linewidth=0.5)

    def updateSat(self, t, sat_plots):

        for i, sat in enumerate(self.satellites):
            
            # Get coordinates of satellite at current time
            x, y, z =  self.satellites[i].at(t).position.km
            
            # Update satellite plot's position on graph
            sat_plots[i].set_data([x], [y])
            sat_plots[i].set_3d_properties([z])
            

    def addSatellite(self, TLE):

        # Add satellite to the list
        self.satellites.append(EarthSatellite(TLE.line1, TLE.line2, name=TLE.name))

    def animate(self, save=False):

        # Ensure the environment is initialized for animation
        if not self.duration:
            raise RuntimeError("Cannot animate without a duration.")

        # Generate time array
        t = self.ts.linspace(self.start_time, self.end_time, self.duration*100)
        
        sat_plots = []

        for i, sat in enumerate(self.satellites):

            # Plot orbital path
            pos = sat.at(t).position.km
            x, y, z = pos
            self.ax.plot(x, y, z)

            # Create plot for satellite
            sat_plot, = self.ax.plot(0, 1, marker="o")
            sat_plots.append(sat_plot)
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, self.updateSat, interval=70, repeat=True, 
                                                      frames=t, fargs=(sat_plots,), save_count=3000)

        # Save the animation if desired
        if save:
            f = r"c://Users/User/Desktop/animation.gif" 
            writergif = animation.PillowWriter(fps=15) 
            anim.save(f, writer=writergif)

        # Show the animation
        plt.show()


    def debug(self):
        plt.show()

