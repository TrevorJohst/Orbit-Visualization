from skyfield.api import EarthSatellite
from skyfield.api import load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from datetime import timedelta
import numpy as np

class Environment:
    class _Satellite:
        def __init__(self, EarthSatellite, name, collider):
            """
            Initializes a meta satellite used to better organize satellite data

            Args:
            EarthSatellite - an EarthSatellite object that the satellite is based around
            name - norad ID of the satellite used to refer to it
            collider - collider object used in close encounter handling (optional)
            """

            # Object variables
            self.sat = EarthSatellite
            self.name = name
            self.collider = collider
            self.plot = None

    def __init__(self, env_radius, start_time, duration=None, grid=True, darkmode=False, earth=True):
        """
        Initializes an environment for orbits to be represented in

        Args:
        env_radius - half of the total width of the graph environment in km, should be >6371
        start_time - when the simulation should be started, a time object from the skyfield library
        duration - length of simulation in hours, can be decimal (Optional)
        grid - whether or not to display the 3D grid (Defaults to True)
        darkmode - whether or not to display output in darkmode (Defaults to False)
        earth - whether or not to display the Earth in animation (Defaults to True)
        """

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

        # Create figure and set sizing
        self.fig = plt.figure(figsize = (7,6))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.ax1 = self.fig.add_subplot(gs[0], projection = '3d')
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

        if earth:
            self.ax1.plot_wireframe(x, y, z, color=linecolor, linewidth=0.5)
    
    def addSatellite(self, TLE_line1, TLE_line2, collider=None):
        """
        Adds one satellite to the list along with its name

        Args:
        TLE_line1, TLE_line2 - corresponding lines of the TLE
        collider - collider object for this satellite (optional)
        """

        # Build data for constructing the satellite object
        EarthSat = EarthSatellite(TLE_line1, TLE_line2)
        name = TLE_line1.split(' ')[1]

        # Append a satellite object to the list
        self.satellites.append(self._Satellite(EarthSat, name, collider))

    def collisionEllipsoid(self, sat_meta, time):
        """
        Creates a collision ellipsoid along the in-track and returns it

        Args:
        sat_meta - meta satellite object we are tailoring the ellipsoid to
        time - SkyField time object of time at desired ellipsoid

        Returns:
        Tuple of ndarrays for plotting an ellipsoid surface (X, Y, Z)
        """

        # Unpack data from meta satellite
        in_track = sat_meta.collider.in_track
        cross_track = sat_meta.collider.cross_track
        radial = sat_meta.collider.cross_track

        # Create base ellipsoid data
        theta, phi = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        
        x = in_track * np.sin(theta)*np.cos(phi)
        y = cross_track * np.sin(theta)*np.sin(phi)
        z = radial * np.cos(theta)

        # Determine angles of velocity vector
        vx, vy, vz = sat_meta.sat.at(time).velocity.m_per_s
        speed0 = np.sqrt(vx**2 + vy**2 + vz**2)
                        
        upsilon = np.arctan(vy/vx)
        if vx < 0:
            beta = -np.arccos(vz/speed0) + np.pi/2
        else:
            beta = np.arccos(vz/speed0) - np.pi/2
        
        # Z-axis rotation matrix
        Rz = np.array([[np.cos(upsilon),    -np.sin(upsilon),     0],
                       [np.sin(upsilon),    np.cos(upsilon),      0],
                       [0,                  0,                    1]])

        # Y-axis rotation matrix
        Ry = np.array([[np.cos(beta),       0,          np.sin(beta)],
                       [0,                  1,          0           ],
                       [-np.sin(beta),      0,          np.cos(beta)]])

        # TODO: X-axis rotation matrix(?)
        
        # Rotate all ellipsoid points
        OX = []
        OY = []
        OZ = []

        for i in range(len(x)):
            pos = np.array([x[i], y[i], z[i]])
            A = Rz @ Ry @ pos # There has to be a better way to do this

            OX.append(A[0])
            OY.append(A[1])
            OZ.append(A[2])

        return (np.asarray(OX), np.asarray(OY), np.asarray(OZ))

    def animate(self, file_name=None, comparison=False, scroll_graph=False, colliders=False):
        """
        Produces an animation of existing satellites in orbit (environment must have a duration)

        Args:
        file_name - Location and file name to save animation at (EX: c:\Documents\Orbits\Starlink-4171)
        comparison - Whether or not to include the separation graph, must have exactly 2 satellites (Defaults to False)
        scroll_graph - Whether or not the comparison graph should scroll with data or display all at once (Defaults to False)
        colliders - Whether or not to display colliders when a close approach occurs (Defaults to False)
        """

        # Ensure the environment is initialized for animation
        if not self.duration:
            raise RuntimeError("Cannot animate without a duration.")

        # If we are including the comparison animation ensure we have the correct number of satellites
        if comparison and len(self.satellites) != 2:
            raise RuntimeError("Cannot compare without exactly 2 satellites.")

        # If colliders are enabled ensure both satellites have collider objects
        if colliders and (not self.satellites[0].collider or not self.satellites[1].collider):
            raise RuntimeError("Colliders was enabled but one of the satellites lacks a collider object")
        
        def update(time):
            """
            Helper method that updates our environment

            Args:
            time - SkyField time object of time at desired frame
            """

            for sat_meta in self.satellites:
                
                # Get coordinates of satellite at current time
                x, y, z =  sat_meta.sat.at(time).position.km
                
                # Update satellite plot's position on graph
                sat_meta.plot.set_data([x], [y])
                sat_meta.plot.set_3d_properties([z])

            if comparison:

                # Find separation of satellites
                x0, y0, z0 = self.satellites[0].sat.at(time).position.km
                x1, y1, z1 = self.satellites[1].sat.at(time).position.km
                dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

                # Use parent method's arrays
                nonlocal separations
                nonlocal times

                # Append separation distance and current time onto arrays
                separations = np.append(separations, dist)
                
                if colliders:

                    # Remove old colliders if needed
                    nonlocal collider0
                    if collider0.axes:
                        collider0.remove()
                    
                    nonlocal collider1
                    if collider1.axes:
                        collider1.remove()

                    # Close approach handling
                    if dist <= 1000:

                        # Create collision ellipsoids with passed in parameters
                        ellipsoid0 = self.collisionEllipsoid(self.satellites[0], time)
                        ellipsoid1 = self.collisionEllipsoid(self.satellites[1], time)
                        
                        collider0 = self.ax1.plot_surface(ellipsoid0[0] + x0, ellipsoid0[1] + y0, ellipsoid0[2] + z0, color='r', alpha=0.2)
                        collider1 = self.ax1.plot_surface(ellipsoid1[0] + x1, ellipsoid1[1] + y1, ellipsoid1[2] + z1, color='r', alpha=0.2)

                # Increment time
                if times.size == 0:
                    times = np.append(times, 0)
                else:
                    times = np.append(times, times[-1] + 0.5) # 120 samples per hour, 0.5 min per sample

                # Set x and y data to updated arrays
                line.set_xdata(times)
                line.set_ydata(separations)

                # Scroll both x and y axes to accommodate new data
                if scroll_graph:
                    length = len(separations)/2

                    if length > repeat:
                        self.ax2.set_xlim(length-repeat, length)
                    else:
                        self.ax2.set_xlim(0, repeat)

                self.ax2.set_ylim(0 - 500, np.amax(separations) + 500)

        # Create local variables
        t = self.ts.linspace(self.start_time, self.end_time, int(self.duration*120))
        if comparison:
            separations = np.empty(0)
            times = np.empty(0)
            repeat = 60
            line, = self.ax2.plot([], [])

            # Make second graph visible
            self.ax2.set_visible(True)
            self.ax2.set_xlim(0, self.duration*60)

        for sat_meta in self.satellites:
            
            # Create plot for satellite
            sat_plot, = self.ax1.plot(0, 1, marker="o")
            sat_meta.plot = sat_plot

            # Plot orbital path
            pos = sat_meta.sat.at(t).position.km
            x, y, z = pos
            self.ax1.plot(x, y, z, label=sat_meta.name, color=sat_plot.get_color())    
            
        if colliders:

            # Create empty collider surfaces
            collider0 = self.ax1.plot_surface(np.empty((0,0)), np.empty((0,0)), np.empty((0,0)))
            collider1 = self.ax1.plot_surface(np.empty((0,0)), np.empty((0,0)), np.empty((0,0)))

        # Show legend
        self.ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0, labelcolor='linecolor')
            
        # Create animation
        anim = animation.FuncAnimation(self.fig, update, interval=70, repeat=False, 
                                                      frames=t, save_count=3000)

        # Save the animation if desired
        if file_name:
            f = file_name + ".gif"
            writergif = animation.PillowWriter(fps=15) 
            anim.save(f, writer=writergif)
            
        # Show the animation
        plt.show()

    def image(self):
        """
        Produces an image of existing satellites in orbit
        """
        
        t = self.ts.linspace(self.start_time, self.end_time, self.duration*120)

        for sat_meta in self.satellites:

            # Plot orbital path
            pos = sat_meta.sat.at(t).position.km
            x, y, z = pos
            self.ax1.plot(x, y, z, label=sat_meta.name)
            
        # Show legend
        self.ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0, labelcolor='linecolor')

        # Show the graph
        plt.show()

