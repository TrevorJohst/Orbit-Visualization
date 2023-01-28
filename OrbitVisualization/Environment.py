from pprint import pp
from skyfield.api import EarthSatellite
from skyfield.api import load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from datetime import timedelta
import numpy as np

class Environment:
    def __init__(self, env_radius, start_time, duration=None, grid=False, darkmode=True):
        """
        Initializes an environment for orbits to be represented in

        Args:
        env_radius - half of the total width of the graph environment, should be >6371
        start_time - when the simulation should be started, a time object from the skyfield library
        duration - length of simulation in hours (Optional)
        grid - whether or not to display the 3D grid (Defaults to False)
        darkmode - whether or not to display output in darkmode (Defaults to True)
        """

        # Object variables
        self.satellites = []
        self.names = []
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

        # Create figure and set sizing
        self.fig = plt.figure(figsize = (7,6))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.ax1 = self.fig.add_subplot(gs[0],projection = '3d')
        self.ax2 = self.fig.add_subplot(gs[1])

        # Configure second graph
        self.ax2.spines[['right', 'top']].set_visible(False)
        self.ax2.set_xlabel('Time (min)')
        self.ax2.set_ylabel('Separation (km)')

        # Hide it until required
        self.ax2.set_visible(False)

        # Reduce number of tick labels
        self.ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax1.zaxis.set_major_locator(plt.MaxNLocator(3))

        # Set visual factors
        self.ax1.set_box_aspect([1,1,0.9])
        self.ax1.view_init(10, 45)

        # Set size of graph
        self.ax1.set_xlim(-env_radius, env_radius)
        self.ax1.set_ylim(-env_radius, env_radius)
        self.ax1.set_zlim(-env_radius, env_radius)

        # Remove axis panes for cosmetic reasons
        self.ax1.xaxis.pane.fill = False
        self.ax1.yaxis.pane.fill = False
        self.ax1.zaxis.pane.fill = False
        
        # Grid handling
        if not grid:
            self.ax1.axis('off')

        # Darkmode handling
        if darkmode:   
            linecolor = 'white'

            # Set backgrounds to dark color
            self.fig.patch.set_facecolor(dark)
            self.ax1.set_facecolor(dark)
            self.ax2.set_facecolor(dark)
            
            # Set 3D graph's axes to white
            self.ax1.w_xaxis.line.set_color('white')
            self.ax1.w_yaxis.line.set_color('white')
            self.ax1.w_zaxis.line.set_color('white')

            # Set 2D graph's labels to white
            self.ax2.xaxis.label.set_color('white')
            self.ax2.yaxis.label.set_color('white')   
            
            # Set 2D graph's axes to white
            self.ax2.spines['left'].set_color('white')
            self.ax2.spines['bottom'].set_color('white')

            # Set both graph's ticks to white
            self.ax1.tick_params(colors='white')
            self.ax2.tick_params(colors='white')
            
        # Generate and graph sphere
        theta, phi = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]

        radius = 6371 # Radius of the Earth in km

        x = radius*np.cos(theta)*np.sin(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(phi)

        self.ax1.plot_wireframe(x, y, z, color=linecolor, linewidth=0.5)

    def addSatellite(self, TLE):
        """
        Adds one satellite to the list along with its name

        Args:
        TLE - TLE object for desired satellite
        """

        # Add satellite to the list
        self.satellites.append(EarthSatellite(TLE.line1, TLE.line2))
        
        # Add name to the list
        self.names.append(TLE.line1.split(' ')[1])

    def animate(self, filename=None, comparison=False):
        """
        Produces an animation of existing satellites in orbit (environment must have a duration)

        Args:
        filename - Location and filename to save animation at, starts at base directory (EX: Documents\Orbits\Starlink-4171)
        comparison - Whether or not to include the separation graph (environment must have exactly 2 satellites)
        """

        # Ensure the environment is initialized for animation
        if not self.duration:
            raise RuntimeError("Cannot animate without a duration.")

        # If we are including the comparison animation ensure we have the correct number of satellites
        if comparison and len(self.satellites) != 2:
            raise RuntimeError("Cannot compare without exactly 2 satellites.")
        
        def update(time):

            for i, sat in enumerate(self.satellites):
                
                # Get coordinates of satellite at current time
                x, y, z =  self.satellites[i].at(time).position.km
                
                # Update satellite plot's position on graph
                sat_plots[i].set_data([x], [y])
                sat_plots[i].set_3d_properties([z])

            if comparison:

                # Find separation of satellites
                x0, y0, z0 = self.satellites[0].at(time).position.km
                x1, y1, z1 = self.satellites[1].at(time).position.km
                dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
                
                # Use parent method's arrays
                nonlocal separations
                nonlocal times

                # Append separation distance and current time onto arrays
                separations = np.append(separations, dist)

                if times.size == 0:
                    times = np.append(times, 0)
                else:
                    times = np.append(times, times[-1] + 0.5) # 120 samples per hour, 0.5 min per sample

                # Set x and y data to updated arrays
                line.set_xdata(times)
                line.set_ydata(separations)

                # Scroll both x and y axes to accommodate new data
                length = len(separations)/2

                if length > repeat:
                    self.ax2.set_xlim(length-repeat, length)
                else:
                    self.ax2.set_xlim(0, repeat)

                self.ax2.set_ylim(0 - 500, np.amax(separations) + 500)

        # Create local variables
        t = self.ts.linspace(self.start_time, self.end_time, self.duration*120)
        sat_plots = []
        if comparison:
            separations = np.empty(0)
            times = np.empty(0)
            repeat = 60
            line, = self.ax2.plot([], [])

            # Make second graph visible
            self.ax2.set_visible(True)

        for i, sat in enumerate(self.satellites):

            # Plot orbital path
            pos = sat.at(t).position.km
            x, y, z = pos
            self.ax1.plot(x, y, z, label=self.names[i])

            # Create plot for satellite
            sat_plot, = self.ax1.plot(0, 1, marker="o")
            sat_plots.append(sat_plot)           
            
        # Show legend
        self.ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0, labelcolor='linecolor')
            
        # Create animation
        anim = animation.FuncAnimation(self.fig, update, interval=70, repeat=False, 
                                                      frames=t, save_count=3000)

        # Save the animation if desired
        if filename:
            f = r"C:\\" + filename + ".gif"
            writergif = animation.PillowWriter(fps=15) 
            anim.save(f, writer=writergif)

        # Show the animation
        plt.show()

    def image(self):
        """Produces an image of existing satellites in orbit"""
        
        t = self.ts.linspace(self.start_time, self.end_time, self.duration*120)

        for i, sat in enumerate(self.satellites):

            # Plot orbital path
            pos = sat.at(t).position.km
            x, y, z = pos
            self.ax1.plot(x, y, z, label=self.names[i])
            
        # Show legend
        self.ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0, labelcolor='linecolor')

        # Show the graph
        plt.show()

