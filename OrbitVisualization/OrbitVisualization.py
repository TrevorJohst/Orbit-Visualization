from Environment import Environment
from skyfield.api import load

class TLE:
    def __init__(self, TLE):

        # Extract TLE lines according to file formatting
        lines = TLE.splitlines()

        if len(lines) == 3:
            self.name = lines[0]
            self.line1 = lines[1]
            self.line2 = lines[2]
        elif len(lines) == 2:
            self.name = None
            self.line1 = lines[0]
            self.line2 = lines[1]
        else:
            raise ValueError

    def __str__(self):

        # Build output string
        output = ""

        if self.name:
            output += self.name + "\n"
        output += self.line1 + "\n"
        output += self.line2

        return output


if __name__ == "__main__":
    test = TLE("""NEXTSAT-1
1 43811U 18099BF  23017.44801183  .00004536  00000+0  39757-3 0  9997
2 43811  97.5949  83.7508 0011279 267.0241  92.9691 14.97097622224635""")
    test2 = TLE("""STARLINK-3106
1 49164U 21082AL  23017.70091959  .00023448  00000+0  19664-2 0  9998
2 49164  70.0020 286.4184 0003135 278.2144  81.8659 14.98329354 75836""")
    
    ts = load.timescale()
    
    Earth = Environment(6371, 8000, ts.utc(2023, 1, 10), duration=2, grid=False, darkmode=True)
    Earth.addSatellite(test)
    Earth.addSatellite(test2)
    Earth.animate(save=True)