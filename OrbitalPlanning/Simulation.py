import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

class Body:
    """
    A simple struct for storing simulation parameters for a body in space.
    """
    def __init__(self, 
                 name: str, 
                 position: tuple[float, float], 
                 velocity: tuple[float, float],
                 color: str = None
                ) -> None:
        """
        Args:
            name (str): Name of the body (e.g., "Earth")
            position ((float, float)): Initial position (x, y) in AU
            velocity ((float, float)): Initial velocity (vx, vy) in AU/day
            color (str): Color for plotting in matplotlib, auto generated if None
        """
        self.name = name
        self.position = position
        self.velocity = velocity
        self.color = color

class Orbit:
    """
    A simple class for storing and operating on a parameterized orbit.
    """
    def __init__(self, 
                 orbit: np.ndarray, 
                 MU: float
                 ) -> None:
        """
        Args:
            orbit (np.ndarray): Orbit matrix of shape 4xT where 4 is the number of state
            variables (x, y, vx, vy) and T is the number of sampled times
            MU (float): Gravitational mass constant G * M
        """
        self.MU = MU
        self._orbit = orbit
        self.update(orbit)

    def update(self, 
               orbit: np.ndarray
               ) -> None:
        """
        Updates the orbit stored in this object.

        Args:
            orbit (np.ndarray): Orbit matrix of shape 4xT where 4 is the number of state
            variables (x, y, vx, vy) and T is the number of sampled times
        """
        self.radii_vec = [orbit[0:2, i] for i in range(orbit.shape[1])]
        self.radii = np.linalg.norm(orbit[0:2, :], axis=0)
        self.velocities_vec = [orbit[2:4, i] for i in range(orbit.shape[1])]
        self.velocities = np.linalg.norm(orbit[2:4, :], axis=0)
        self.periapse_idx = np.argmax(self.velocities)
        self.apoapse_idx = np.argmin(self.velocities)
        self.energy = 1/2 * self.velocities[0]**2 - self.MU / self.radii[0]

    @property
    def orbit(self) -> np.ndarray:
        return self._orbit

    @orbit.setter
    def orbit(self, value: np.ndarray) -> None:
        self._orbit = value
        self.update(value)

class SunFixedSimulator:
    """
    A simple simulation assuming the Sun is fixed at the center of the system and dominates physical effects.
    """

    def __init__(self,
                 bodies: list[Body],
                 max_step: float = 1.0,
                 rtol: float = 1e-6,
                 atol: float = 1e-6
                ) -> None:
        """
        Args:
            bodies ([Body]): List of celestial bodies in the simulation environment
            max_step (float): Maximum step size for numerical integration in days
            rtol (float): Relative tolerance for scipy.integrate.solve_ivp
            atol (float): Absolute tolerance for scipy.integrate.solve_ivp
        """
        self.bodies = bodies

        # Physics constants
        self.AU = 1.496e11  # m
        self.DAY = 24*60*60 # s
        G = 6.673e-11       # N m**2 kg**-2
        m_sun = 1.989e30    # kg
        self.MU_SUN = G * m_sun * self.DAY**2 / self.AU**3  # AU**3 DAY**-2

        # Simulation constants
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol

    def simulateOrbits(self, 
                       tspan: tuple[float, float],
                       current_state: dict = None,
                       bodies: list[Body] = None
                      ) -> tuple[np.ndarray, dict]:
        """
        Simulates the orbits of all bodies in the simulator.

        Args:
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days)
            current_state (dict): Dictionary mapping each body to a state (x, y, vx, vy), if
            passed will override the initial conditions and simulate starting from this state
            bodies ([Body]): List of bodies to simulate the orbits for, if passed will only 
            simulate these bodies, ignoring others in the system

        Returns:
            tuple: A tuple of the time and states as simulated
                - np.ndarray: Times the orbit is simulated at
                - dict: Dictionary mapping each body to its corresponding orbit
        """
        # Use system bodies if not overriden
        if not bodies: bodies = self.bodies

        def orbitalEquations(t, s):
            """
            Orbital equations in differential form. s has the form (x, y, vx, vy) * i
            where i is the number of bodies in the simulation.
            """
            n_bodies = len(bodies)
            dsdt = []

            for i in range(n_bodies):
                # Extract current state of this body
                b_x  = s[i * 4]
                b_y  = s[i * 4 + 1]
                b_vx = s[i * 4 + 2]
                b_vy = s[i * 4 + 3]

                # dx/dt = vx
                dsdt.append(b_vx)

                # dy/dt = vy
                dsdt.append(b_vy)

                # Distance from sun
                r = np.sqrt(b_x**2 + b_y**2)  

                # Acceleration due to gravity
                ax = -self.MU_SUN * b_x / r**3
                ay = -self.MU_SUN * b_y / r**3

                # dvx/dt = ax
                dsdt.append(ax)

                # dvy/dt = ay
                dsdt.append(ay)

            return np.array(dsdt)

        # Extract initial conditions
        init_conditions = []
        if current_state:
            for body_state in current_state.values():
                init_conditions.extend(body_state)
        else:
            for body in bodies:
                init_conditions.extend(body.position + body.velocity)

        # Integrate using solve_ivp
        solution = solve_ivp(
            orbitalEquations, 
            tspan, 
            init_conditions,
            rtol=self.rtol, 
            atol=self.atol,
            max_step=self.max_step
        )

        # Store resulting states in a dictionary
        states = {
            body.name: Orbit(solution.y[i * 4:(i + 1) * 4, :], self.MU_SUN)
            for i, body in enumerate(bodies)
        }

        return solution.t, states
    
    def hohmannTransfer(self, 
                        orbit1: Orbit, 
                        orbit2: Orbit,
                        sim_full: bool = False
                        ) -> Orbit:
        """
        Calculates the Hohmann transfer orbit to move between orbit 1 and orbit 2.

        Args:
            orbit1 (Orbit): Orbit object for the origin orbit
            orbit2 (Orbit): Orbit object for the destination orbit
            sim_full (bool): If True, the full transfer orbit is returned rather than half

        Returns:
            Orbit: Orbit matrix for the transfer orbit
        """

        # Semi-major axis of the transfer orbit
        a = (orbit1.radii[orbit1.periapse_idx] + orbit2.radii[orbit2.apoapse_idx]) / 2

        # Velocity at periapsis for orbit 1 (before the burn)
        v1 = orbit1.velocities[orbit1.periapse_idx]

        # Periapsis velocity in the transfer orbit (after the burn)
        r_peri_vec = orbit1.radii_vec[orbit1.periapse_idx]
        r_peri = orbit1.radii[orbit1.periapse_idx]
        v2 = np.sqrt(self.MU_SUN * (2 / r_peri - 1 / a))

        # Velocity required to change apoapsis to radius of orbit 2
        delta_v = v2 - v1

        # Add delta v in tangential direction of orbit
        tangential_unit = orbit1.velocities_vec[orbit1.periapse_idx] / v1
        v_new = orbit1.velocities_vec[orbit1.periapse_idx] + delta_v * tangential_unit

        # Create a satellite ready to make the transfer
        satellite = Body(
            name="Satellite",
            position=(r_peri_vec[0], r_peri_vec[1]), 
            velocity=(v_new[0], v_new[1])
        )

        # Transfer time is half of transfer orbit's period (unless we want full orbit)
        t_transfer = np.pi * np.sqrt(a**3 / self.MU_SUN)
        if sim_full: t_transfer *= 2

        return self.simulateOrbits(tspan=(0, t_transfer), bodies=[satellite])[1]["Satellite"]
       
    def plotOrbits(self, 
                   tspan: tuple[float, float]
                   ) -> None:
        """
        Plot the orbits of all bodies currently in the simulator.

        Args:
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days)
        """
        # Simulate the orbits
        t, states = self.simulateOrbits(tspan)

        # Plot each body's orbit in AU
        for body in self.bodies:
            plt.plot(states[body.name].orbit[0], states[body.name].orbit[1], c=body.color, label=body.name)

        plt.legend()
        plt.grid(alpha=0.5)
        plt.xlabel('X position (AU)')
        plt.ylabel('Y position (AU)')
        plt.title('Orbit Simulation')
        plt.axis('equal')
        plt.show()

    def plotHohmannTransfer(self,
                            tspan: tuple[float, float],
                            transfer_map: tuple[int | str]
                            ) -> None:
        """
        Plots the orbits of all bodies in the simulator, and a Hohmann transfer
        orbit between two selected orbits. Calculations may fail if the simulation
        timespan is not sufficient to cover the full orbits.

        Args:
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days)
            transfer_map ((int | str)): Tuple mapping two body indices or names together
            in the order they should be transferred from (e.g. (0, 2) or ('Earth', 'Mars'))
        """
        # Simulate the orbits
        t, states = self.simulateOrbits(tspan)

        # Plot each body's orbit in AU
        for body in self.bodies:
            plt.plot(states[body.name].orbit[0], states[body.name].orbit[1], c=body.color, label=body.name, zorder=5)

        # Determine what bodies to transfer between the orbits of
        if isinstance(transfer_map[0], str):
            body1 = next((body.name for body in self.bodies if body.name == transfer_map[0]), None)
            body2 = next((body.name for body in self.bodies if body.name == transfer_map[1]), None)

            if not body1 or not body2:
                raise RuntimeError("Invalid transfer map, one or more bodies not found")
        
        elif isinstance(transfer_map[0], int):
            body1 = self.bodies[transfer_map[0]].name
            body2 = self.bodies[transfer_map[1]].name

        else:
            raise RuntimeError("Invalid transfer map, must be a string or integer")
        
        # Simulate the Hohmann transfer
        hohmann_orbit = self.hohmannTransfer(states[body1], states[body2])

        # Plot the Hohmann transfer
        plt.plot(hohmann_orbit.orbit[0], hohmann_orbit.orbit[1], label="Satellite", c='black')

        plt.legend()
        plt.grid(alpha=0.5)
        plt.xlabel('X position (AU)')
        plt.ylabel('Y position (AU)')
        plt.title(f'Hohmann Transfer: {body1} -> {body2}')
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    
    earth = Body(
        name="Earth",
        position=(1, 0),
        velocity=(0, 0.0172),
        color='slateblue'
    )
    mars = Body(
        name="Mars",
        position=(1.52, 0),
        velocity=(0, 0.0139),
        color='firebrick'
    )

    # Create Simulator instance with bodies
    simulator = SunFixedSimulator([earth, mars])
    tspan = (0, 700) 

    # Plot results
    # simulator.plotOrbits(tspan)
    simulator.plotHohmannTransfer(tspan, ('Mars', 'Earth'))