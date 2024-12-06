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
    A simple class for storing and operating on a samled orbit.
    """
    def __init__(self, 
                 orbit: np.ndarray, 
                 t: np.ndarray,
                 MU: float
                 ) -> None:
        """
        Args:
            orbit (np.ndarray): Orbit matrix of shape 4xT where 4 is the number of state
                variables (x, y, vx, vy) and T is the number of sampled times
            t (np.ndarray): Times that the orbit is simulated for, shape 1xT
            MU (float): Gravitational mass constant G * M
        """
        self.MU = MU
        self.t = t
        self.orbit = orbit

        # First two states are position
        self.radii_vec = [orbit[0:2, i] for i in range(orbit.shape[1])]
        self.radii = np.linalg.norm(orbit[0:2, :], axis=0)

        # Second two states are velocity
        self.velocities_vec = [orbit[2:4, i] for i in range(orbit.shape[1])]
        self.velocities = np.linalg.norm(orbit[2:4, :], axis=0)

        # Periapse is at max velocity, apoapse is at min velocity
        self.periapse_idx = np.argmax(self.velocities)
        self.apoapse_idx = np.argmin(self.velocities)

        # True anomalies for every point
        periapse_vector = self.radii_vec[self.periapse_idx]
        self.true_anomalies = np.array([
            np.arctan2(
                r[..., 0] * periapse_vector[..., 1] - r[..., 1] * periapse_vector[..., 0], # ||r x rp||
                np.dot(r, periapse_vector))                                                # r * rp
            for r in self.radii_vec
        ])

        # Period if the orbit is complete
        angle_diffs = (np.diff(self.true_anomalies) + np.pi) % (2 * np.pi) - np.pi
        angle_sums = np.cumsum(np.abs(angle_diffs))
        if angle_sums[-1] >= (2*np.pi - 0.1):
            complete_orbit_idx = np.argmin(np.abs(angle_sums - 2*np.pi))
            self.period = self.t[complete_orbit_idx] - self.t[0]
        else:
            self.period = None

        # Vis-viva equation
        self.energy = 1/2 * self.velocities[0]**2 - self.MU / self.radii[0]

        # Semi-major axis
        self.a = (self.radii[self.periapse_idx] + self.radii[self.apoapse_idx]) / 2

    def when(self,
             query_time: float
             ) -> int:
        """
        Returns the index into the orbit closest to a given time of interest
        """
        return np.argmin(abs(self.t - query_time))
    
    def where(self,
              query_loc: np.ndarray | float,
              angle: bool = False
              ) -> int:
        """
        Returns the index into the orbit closest to a given location of interest (either location or anomaly)    
        """
        if angle: return np.argmin(abs(self.true_anomalies - query_loc))
        else:     return np.argmin(np.linalg.norm(self.radii_vec - query_loc))

    def minSeparation(self,
                       orbit2: "Orbit",
                      ) -> tuple[float, int, int]:
        """
        Computes the minimum separation distance between two orbits

        Args:
            orbit2 (Orbit): Another orbit to compute the distance between

        Returns:
            (dist, this_idx, that_idx) ((float, int, int)):
            A tuple of form (dist, this_idx, that_idx), where dist
            is the separation distance, and this_idx, that_idx are the indices in this 
            and that orbit where the nearest approach occurs
        """
        squared_distances = np.sum(np.pow(self.orbit[0:2, :, np.newaxis] - orbit2.orbit[0:2, np.newaxis, :], 2), axis=0)
        index1, index2 = np.unravel_index(np.argmin(squared_distances), squared_distances.shape)
        return np.sqrt(squared_distances[index1, index2]), index1, index2
    
    def findIntercepts(self,
                       orbit2: "Orbit"
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the intercepts between two orbits if it exists.

        Args:
            orbit2 (Orbit): Another orbit to find the intercept with

        Returns:
            (this_idx, that_idx) ((np.ndarray, np.ndarray)): 
            A tuple of the indices in each orbit where the intercepts occur, 
            if an intercept does not occur, each return is None. First array is for this orbit,
            second array is for the passed orbit.
        """
        self_radii = np.array(self.radii_vec)
        orbit2_radii = np.array(orbit2.radii_vec)
        distances = np.linalg.norm(self_radii[:, np.newaxis, :] - orbit2_radii[np.newaxis, :, :], axis=2)
        indices = np.argwhere(distances < 0.01)
        if indices.size > 0:
            return indices[0, 0], indices[0, 1]
                
        return None, None

class SunFixedSimulator:
    """
    A simple simulation assuming the Sun is fixed at the center of the system and dominates physical effects.
    """

    def __init__(self,
                 bodies: list[Body],
                 max_step: float = 1.0,
                 max_sim: int = 2000,
                 rtol: float = 1e-6,
                 atol: float = 1e-6
                 ) -> None:
        """
        Args:
            bodies ([Body]): List of celestial bodies in the simulation environment
            max_step (float): Maximum step size for numerical integration in days
            max_sim (int): Maximum number of days to simulate for if not specified by user
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
        self.max_sim = max_sim
        self.rtol = rtol
        self.atol = atol

    def simulateOrbit(self,
                      body: Body,
                      tspan: tuple[float, float] = None,
                      current_state: tuple[float, float, float, float] = None
                      ) -> Orbit:
        """
        Simulates the orbit of a single body.

        Args:
            body (Body): The body to simulate the orbit for
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days), if not passed 
                simulates one complete orbit
            current_state ((float, float, float, float)): Tuple describing the current state 
                of the body (x, y, vx, vy), if passed will override the initial conditions

        Returns:
            Orbit: The simulated orbit
        """
        def orbitalEquations(t, s):
            """
            Orbital equations in differential form. s has the form (x, y, vx, vy)
            """
            # Extract current state of this body
            x, y, vx, vy = s

            # Distance from sun
            r = np.sqrt(x**2 + y**2)

            # Differential form of equations
            return np.array([
                vx,                         # dx/dt
                vy,                         # dy/dt
                -self.MU_SUN * x / r**3,    # dvx/dt
                -self.MU_SUN * y / r**3     # dvy/dt
            ])
        
        def completeOrbit(t, s):
            """
            Terminates the simulation if a complete orbit is reached
            """
            # Never terminate within the first few days to avoid premature termination
            if t < 2: return 1
            x, y, vx, vy = s
            r_current = np.sqrt((x - init_conditions[0])**2 + (y - init_conditions[1])**2)
            return r_current - (self.atol*1e4)
        
        # Initial conditions
        init_conditions = []
        if current_state: init_conditions.extend(current_state)
        else:             init_conditions.extend(body.position + body.velocity)

        # Integrate using solve_ivp normally if tspan is passed
        if tspan:
            solution = solve_ivp(
                orbitalEquations, 
                tspan, 
                init_conditions,
                rtol=self.rtol, 
                atol=self.atol,
                max_step=self.max_step
            )

        # Terminate integration when a complete orbit is detected otherwise
        else:
            completeOrbit.terminal = True
            completeOrbit.direction = 0
            solution = solve_ivp(
                orbitalEquations, 
                (0, self.max_sim), 
                init_conditions,
                rtol=self.rtol, 
                atol=self.atol,
                max_step=self.max_step,
                events=completeOrbit
            )

        return Orbit(solution.y, solution.t, self.MU_SUN)

    def simulateOrbits(self, 
                       bodies: list[Body] = None,
                       tspan: tuple[float, float] = None,
                       current_state: dict = None
                      ) -> dict:
        """
        Simulates the orbits of all bodies in the simulator.

        Args:
            bodies ([Body]): List of bodies to simulate the orbits for, if not passed will
                simulate the orbit of every body in the simulator
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days)
            current_state (dict): Dictionary mapping each body to a state (x, y, vx, vy), 
                if passed will override the initial conditions and simulate 
                starting from this state

        Returns:
            dict: Dictionary mapping each body to its corresponding orbit
        """
        # Use system bodies if not overriden
        if not bodies: bodies = self.bodies

        # Simulate normally if we are given a timespan
        if tspan:
            orbits = dict()
            for body in bodies:
                if current_state:
                    orbit = self.simulateOrbit(body, tspan, current_state[body.name])
                else:
                    orbit = self.simulateOrbit(body, tspan)
                orbits[body.name] = orbit

            return orbits

        # If not given a timespan, simulate each orbit to determine max length needed
        times = []
        for body in bodies:
            if current_state:
                orbit = self.simulateOrbit(body, tspan, current_state[body.name])
            else:
                orbit = self.simulateOrbit(body, tspan)
            times.append(orbit.t)

        # Re-simulate with max time to ensure orbit times align
        max_time = np.max(np.concatenate(times))
        orbits = dict()
        for body in bodies:
            if current_state:
                orbit = self.simulateOrbit(body, (0, max_time), current_state[body.name])
            else:
                orbit = self.simulateOrbit(body, (0, max_time))
            orbits[body.name] = orbit

        return orbits
    
    def hohmannTransfer(self, 
                        orbit1: Orbit, 
                        orbit2: Orbit,
                        sim_full: bool = False
                        ) -> tuple[Orbit, float]:
        """
        Calculates the Hohmann transfer orbit to move between orbit1 and orbit2.

        Args:
            orbit1 (Orbit): Orbit object for the origin orbit
            orbit2 (Orbit): Orbit object for the destination orbit
            sim_full (bool): If True, the full transfer orbit is returned rather than half

        Returns:
            tuple: The transfer orbit and the deltaV required to achieve it
                - Orbit: Orbit object for the transfer orbit
                - float: Magnitude of the deltaV applied
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
        deltaV = v2 - v1

        # Add delta v in tangential direction of orbit
        tangential_unit = orbit1.velocities_vec[orbit1.periapse_idx] / v1
        v_new = orbit1.velocities_vec[orbit1.periapse_idx] + deltaV * tangential_unit

        # Create a satellite ready to make the transfer
        satellite = Body(
            name="Satellite",
            position=(r_peri_vec[0], r_peri_vec[1]), 
            velocity=(v_new[0], v_new[1])
        )

        # Transfer time is half of transfer orbit's period (unless we want full orbit)
        t_start = orbit1.t[orbit1.periapse_idx]
        t_transfer = np.pi * np.sqrt(a**3 / self.MU_SUN)
        if sim_full: t_transfer *= 2

        hohmann_orbit = self.simulateOrbit(body=satellite, tspan=(t_start, t_start + t_transfer))
        return hohmann_orbit, deltaV

    def nonHohmannTransfer(self,
                           orbit: Orbit,
                           anomaly: float,
                           deltaV: float,
                           beta: float,
                           tspan: tuple[float, float] = None
                           ) -> None:
        """
        Executes an arbitrary transfer from the origin orbit.

        Args:
            orbit (Orbit): Origin orbit to transfer from
            anomaly (float): Anomaly within the orbit to initiate transfer (in degrees)
            deltaV (float): Magnitude of the deltaV to apply
            beta (float): Angle from tangent to apply the deltaV (in degrees)

        Returns:
            Orbit: The resulting transfer orbit
        """
        # Wrap angles to [-pi, pi] and find index of transfer
        beta = np.atan2(np.sin(np.deg2rad(beta)), np.cos(np.deg2rad(beta)))
        anomaly = np.atan2(np.sin(np.deg2rad(anomaly)), np.cos(np.deg2rad(anomaly)))
        transfer_idx = orbit.where(anomaly, angle=True)

        # Construct total velocity for transfer
        tangential_unit = orbit.velocities_vec[transfer_idx] / orbit.velocities[transfer_idx]
        R = np.array([[np.cos(beta), -np.sin(beta)], 
                      [np.sin(beta),  np.cos(beta)]])
        v_total = orbit.velocities_vec[transfer_idx] + R @ (deltaV * tangential_unit)

        # Create satellite for transfer and return orbit
        satellite = Body(
            name="Satellite",
            position=(orbit.radii_vec[transfer_idx][0], orbit.radii_vec[transfer_idx][1]),
            velocity=(v_total[0], v_total[1])
        )
        return self.simulateOrbit(satellite, tspan=tspan)

    def plotOrbits(self, 
                   tspan: tuple[float, float] = None
                   ) -> None:
        """
        Plot the orbits of all bodies currently in the simulator.

        Args:
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days),
                if not passed will simulate the full orbit of the longest orbit
        """
        # Simulate the orbits
        orbits = self.simulateOrbits(tspan=tspan)

        # Plot each body's orbit in AU
        for body in self.bodies:
            plt.plot(orbits[body.name].orbit[0], orbits[body.name].orbit[1], c=body.color, label=body.name)

        plt.title('Orbit Simulation')
        self._setupPlot()

    def plotHohmannTransfer(self,
                            transfer_map: tuple[int | str],
                            sim_full: bool = False
                            ) -> None:
        """
        Plots the orbits of all bodies in the simulator, and a Hohmann transfer
        orbit between two selected orbits. Calculations may fail if the simulation
        timespan is not sufficient to cover the full orbits.

        Args:
            transfer_map ((int | str)): Tuple mapping two body indices or names together
                in the order they should be transferred from (e.g. (0, 2) or ('Earth', 'Mars'))
            sim_full (bool): Whether or not to simulate all of the full orbits, or just for the transfer window
        """
        # Parse map
        body1, body2 = self._parseTransferMap(transfer_map)
        bodies = [body for body in self.bodies if body.name == body1 or body.name == body2]

        # Simulate the full orbits
        orbits = self.simulateOrbits(bodies=bodies)
        orbit1 = orbits[body1]
        orbit2 = orbits[body2]
        
        # Simulate the Hohmann transfer
        hohmann_orbit, _ = self.hohmannTransfer(orbit1, orbit2, sim_full)

        # Resimulate the orbits just for the time of the transfer
        if not sim_full:

            # Body1 starts at same time as transfer
            body1_start = orbit1.when(hohmann_orbit.t[0])

            # Calculate when body2 had to be to intercept transfer
            body2_end = orbit2.where(hohmann_orbit.radii_vec[-1])
            body2_start = body2_end - len(hohmann_orbit.t) # Step back this many timesteps

            # Loop back from a complete orbit if we try to index from end
            if body2_start < 0:
                body2_start = orbit2.when(orbit2.period) - len(hohmann_orbit.t)

            orbits = self.simulateOrbits(
                bodies=bodies,
                tspan=(0, len(hohmann_orbit.t)),
                current_state={
                    body1: (
                        orbit1.radii_vec[body1_start][0],
                        orbit1.radii_vec[body1_start][1],
                        orbit1.velocities_vec[body1_start][0],
                        orbit1.velocities_vec[body1_start][1],
                    ),
                    body2: (
                        orbit2.radii_vec[body2_start][0],
                        orbit2.radii_vec[body2_start][1],
                        orbit2.velocities_vec[body2_start][0],
                        orbit2.velocities_vec[body2_start][1],
                    ),
                }
            )

        # Plot each body's orbit
        for body in bodies:
            plt.plot(orbits[body.name].orbit[0], orbits[body.name].orbit[1], c=body.color, label=body.name, zorder=2)

        # Plot the Hohmann transfer
        plt.plot(hohmann_orbit.orbit[0], hohmann_orbit.orbit[1], label="Satellite", c='black', zorder=1)

        # Plot points at the start and end
        plt.scatter(hohmann_orbit.orbit[0, 0], hohmann_orbit.orbit[1, 0], c='black', s=20, zorder=3)
        plt.scatter(hohmann_orbit.orbit[0, -1], hohmann_orbit.orbit[1, -1], c='black', s=20, zorder=3)

        plt.title(f'Hohmann Transfer: {body1} -> {body2}')
        self._setupPlot()

    def _parseTransferMap(self,
                          transfer_map: tuple[int | str]
                          ) -> tuple[str, str]:
        """
        Parses a transfer map between two bodies in the simulation and handles failure.

        Args:
            tspan ((float, float)): Timespan to simulate from t0 to tf (in days)
            transfer_map ((int | str)): Tuple mapping two body indices or names together

        Returns:
            tuple: the string names of both bodies
                - str: body 1
                - str: body 2
        """
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
        
        return body1, body2

    def _setupPlot(self) -> None:
        """
        Does the final setup for the matplot config.
        """
        plt.legend()
        plt.grid(alpha=0.5)
        plt.xlabel('X position (AU)')
        plt.ylabel('Y position (AU)')
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

    # Plot Hohmann Transfers
    simulator.plotHohmannTransfer(('Earth', 'Mars'), False)
    simulator.plotHohmannTransfer(('Mars', 'Earth'), False)

    # Simulate Hohmann Transfer to get deltaV estimate
    orbits = simulator.simulateOrbits()
    _, hohmannV = simulator.hohmannTransfer(orbits['Earth'], orbits['Mars'])

    # Loop through angle perturbations from tangent, investigate intercepts
    intercept_times = []
    betas = []
    for beta in np.arange(-85, 85, 2):
        nonhomann = simulator.nonHohmannTransfer(orbits['Earth'], anomaly=0, deltaV=hohmannV*2, beta=beta)
        transfer_idx, mars_idx = nonhomann.findIntercepts(orbits['Mars'])

        # If perturbed orbit intercepted Mars' orbit, add to results
        if transfer_idx is not None:
            intercept_times.append(nonhomann.t[transfer_idx])
            betas.append(beta)

            plt.plot(nonhomann.orbit[0], nonhomann.orbit[1], c='Black', lw=1, alpha=0.5, zorder=1.5)

    # Plot just the orbits for visualization
    simulator.plotOrbits()

    # Plot our results of the investigation
    plt.scatter(betas, intercept_times, c='green', s=20, label=f"{2*hohmannV*1731.46:.3f} km/s")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel('β (degrees)')
    plt.ylabel('Transfer Time (days)')
    plt.title("Transfer Times")
    plt.show()