from Environment import Environment
from skyfield.api import load
from pathlib import Path

class TLE:
    def __init__(self, TLE):
        """Initialize a TLE object from a string including newlines"""

        # Extract TLE lines according to file formatting
        lines = TLE.splitlines()
        
        if len(lines) == 2:
            self.line1 = lines[0]
            self.line2 = lines[1]
        else:
            raise ValueError

def compareSatellites(file_directory, environment, save_directory=None, colliders=None):
    """
    Produce an animation comparing two satellite orbits

    Args:
    file_directory - location of stored text file assuming cwd, should be 2 TLEs across 4 lines
    environment - an existing environment object that the animation will be produced in
    colliders - Tuple containing all collider details for both satellites (in_track0, cross_track0, radial_0, in_track1, cross_track1, radial_1)
    """
    
    # Unpack data from file
    content = Path(str(Path.cwd()) + file_directory).read_text()
    lines = content.split('\n')
    
    # Ensure the file is valid
    if len(lines) != 4:
        raise RuntimeError("A comparison should contain exactly 2 TLEs.")

    # Add both satellites to environment
    environment.addSatellite(TLE(lines[0] + "\n" + lines[1]))
    environment.addSatellite(TLE(lines[2] + "\n" + lines[3]))

    # Animate the comparison
    environment.animate(filename=save_directory, comparison=True, colliders=colliders)

def graphOrbits(file_directory, environment):
    """
    Produces a graph displaying the orbits of N satellites

    Args:
    file_directory - location of stored text file assuming cwd, should be N TLEs across 2N lines
    environment - an existing environment object that the graph will be produced in
    """
    
    # Unpack data from file
    content = Path(str(Path.cwd()) + file_directory).read_text()
    lines = content.split('\n')

    # Slight sanity check
    if len(lines) % 2 != 0:
        raise RuntimeError("A TLE file should contain an even number of lines.")

    # Add each satellite to the environment
    i = 0
    while i < len(lines):
        environment.addSatellite(TLE(lines[i] + "\n" + lines[i+1]))
        i += 2

    # Produce an image of the orbits
    environment.image()

if __name__ == "__main__":
    
    # File directory of the data, assume current working directory
    # filedirectory = r"\Data\25 Sats\orbits.txt"
    filedirectory = r"\Data\Interesting Collisions\1.txt"

    # Filename to save animation as
    savename = r"Orbit Output\colliders"

    ts = load.timescale()
    test = ts.utc(2023, 1, 11)

    # Earth object our satellites act around
    Earth = Environment(8000, ts.utc(2023, 1, 11), duration=4, grid=False, darkmode=True, Earth=False)

    # Fill collider data (in-track, cross-track, radial)
    colliders = (2000, 1000, 1100,
                 1400, 800, 1000)

    # Compares 2 satellites, output directory can be manually changed
    compareSatellites(filedirectory, Earth, save_directory=None, colliders=colliders)

    # Graphs the orbits of n satellites, file should be a list of TLEs
    #graphOrbits(filedirectory, Earth)