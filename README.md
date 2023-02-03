## Orbit Visualization
A simple tool made in python to help visualize the orbits of satellites around Earth.
  
  
  
## Dependencies
The below three libraries are needed to run this repo
  
* `skyfield` : Implements an SGP4 propagator to simulate the satellite movement
* `matplotlib` : Used for all of the graphing and visualization
* `numpy` : Facilitates the use of linear algebra



## OrbitVisualization.py
This file exists solely to facilitate an easier interaction with the Environment object. It is not required to produce animations or graphs but does implement a few useful methods. Multiple examples are included at the bottom of the file that show how to produce an orbit image, basic animation, and comparison animation.

### loadEnvironment
Will take in a file directory to a .txt file, an environment, and optionally a list of collider objects to streamline the process of loading satellites into the environment. 

The .txt file should contain however many TLEs the user wishes to load with newlines included. Multiple example files are included in the Data folder.

### makeTime
Returns a SkyField time object that can then be used to initialize an environment object. The first three parameters are integer representations of the year, month, and day respectively. The fourth parameter can either be an integer representation of the hour, or a string in the format HH:MM:S (as used by CelesTrak).



## Environment.py
The only file technically required to produce graphs. If interfacing directly with an environment the user will have to do their own file handling and create their own time object.

### Collider
A very basic object that contains the in-track, cross-track, and radial lengths of a colliding ellipsoid. If the user wishes for colliders to be shown in a comparison animation they will need to create one of these objects for each satellite and pass it in when adding the satellite to the environment.

### Environment
The environment that our satellites will act in. When initializing it the user can set the environment size in kilometers, the start time of the simulation (as a SkyField time object), the duration of the simulation in hours, and three other cosmetic options. An example initialization is shown below.

    time = makeTime(2023, 1, 11, '08:06:08.084')
    Earth = Environment(8000, time, duration=2.5, grid=False, darkmode=True, earth=True)
    
### Environment.addSatellite
The method used to add a satellite to the environment. Takes in each line of the TLE as a string, and an optional collider object for this satellite. An example method call is shown below.

    TLE = """1 31789U 07028A   23033.51750760  .00009163  00000+0  38583-3 0  9995
    2 31789  64.4956  78.1033 0059735 281.0419  78.3991 15.22282737860886"""
    lines = TLE.splitlines()
    
    Earth.addSatellite(lines[0], lines[1])
    
### Environment.animate
Will animate the orbits of all satellites in environment assuming a duration was provided when initialized. Can optionally include a file directory to save the animation to and toggle three cosmetic options. An example animation call is shown below.

    file_directory = r"c:\Documents\Orbits\Starlink-4171"
    Earth.animate(file_name=file_directory, scroll_graph=True, comparison=True, colliders=True)
    
An example of a produced animation is also included below.

![colliders](https://user-images.githubusercontent.com/122303295/216702597-5612d3ae-772f-4f16-b1b0-cf347ea1ecf2.gif)

### Environment.image
Produces a still image of the orbits of all satellites currently in the environment. Does not have any parameters, if you wish to save the image you can use the save feature provided in the bottom left of the matplotlib figure. 

An example of a produced image is included below.

![10 Sats](https://user-images.githubusercontent.com/122303295/216703133-96c3df4d-dc52-4992-b8fc-ec0909cef44b.png)

