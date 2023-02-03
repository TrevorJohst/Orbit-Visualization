from Environment import Environment
from Environment import Collider
from skyfield.api import load
from pathlib import Path


def loadEnvironment(file_directory, environment, colliders=None):
    """
    Loads an environment object with satellites from a text file
    
    Args:
    file_directory - location of stored text file assuming cwd, should be N TLEs across 2N lines
    environment - an existing environment object that the graph will be produced in
    colliders - list of collider objects for each satellite
    """
    
    # Unpack data from file
    content = Path(str(Path.cwd()) + file_directory).read_text()
    lines = content.splitlines()

    # Slight sanity check
    if len(lines) % 2 != 0:
        raise RuntimeError("A TLE file should contain an even number of lines.")

    # Add each satellite to the environment
    i = 0
    while i < len(lines):

        if colliders:
            environment.addSatellite(lines[i], lines[i+1], colliders[i // 2])
        else:
            environment.addSatellite(lines[i], lines[i+1])

        i += 2

def makeTime(year, month, day, hour):
    """
    Produces and returns a skyfield time object for use in an environment

    Args:
    year, month, day - integer representation of corresponding time factor
    hour - either an integer or string representation in the format HH:MM:S (as used by CelesTrak)

    Returns:
    Time object corresponding to passed in parameters
    """

    # Make timescale object
    ts = load.timescale()
    
    # Return time object based on what format hour was passed in as
    if type(hour) is int:
        return ts.utc(year, month, day, hour)
    else:
        unpacked = hour.split(':')
        return ts.utc(year, month, day, int(unpacked[0]), int(unpacked[1]), float(unpacked[2]))

if __name__ == "__main__":
    
    # File directory of the data, assume current working directory
    file_directory_orbits = r"\Data\25 Sats\orbits.txt"
    file_directory_comparison = r"\Data\Interesting Collisions\1.txt"

    # File directory to save animation to
    save_directory = r"c:\Orbit Output\\"

    # Name of animation we are saving
    save_name = "colliders"

    # Produce a timescale object for testing
    time = makeTime(2023, 1, 11, '08:06:08.084')

    # Earth object our satellites act around
    Earth = Environment(8000, time, duration=2.5, grid=False, darkmode=False, earth=True)
    

    # # Uncomment to demonstrate loading a file with many satellites and producing an orbital image
    #
    # # Load satellites into environment
    # loadEnvironment(file_directory_orbits, Earth)
    # 
    # # Produce an image of all satellites orbital paths
    # Earth.image()


    # # Uncomment to demonstrate loading a file with many satellites and producing an orbital animation
    # 
    # # Load satellites into environment
    # loadEnvironment(file_directory_orbits, Earth)
    # 
    # # Produce an image of all satellites orbital paths
    # Earth.animate()


    # Uncomment to demonstrate loading a file with two satellites and producing a comparison animation
    
    # Manually create collider objects for our two satellites 
    collider1 = Collider(2000, 1000, 1100)
    collider2 = Collider(1400, 800, 1000)
    
    # Load into a list to be passed into environment
    colliders = [collider1, collider2]
    
    # Load satellites into environment
    loadEnvironment(file_directory_comparison, Earth, colliders)
    
    # Produce an animation of two satellites in orbit comparing their separations and displaying colliders
    Earth.animate(file_name=save_directory + save_name, comparison=True, colliders=True)