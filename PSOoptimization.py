import numpy as np
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100):
        # Store the function to be optimized and its bounds
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # PSO hyperparameters
        self.w = 0.729    # inertia weight - controls influence of previous velocity
        self.c1 = 1.49    # cognitive parameter - controls attraction to personal best
        self.c2 = 1.49    # social parameter - controls attraction to global best
        
        # Initialize particle positions randomly within bounds
        self.positions = np.random.uniform(bounds[:, 0], bounds[:, 1], 
                                         (num_particles, bounds.shape[0]))
        # Initialize velocities as zeros
        self.velocities = np.zeros_like(self.positions)
        
        # Initialize personal best positions and values
        self.pbest = self.positions.copy()
        self.pbest_val = np.array([objective_func(p) for p in self.positions])
        # Initialize global best position and value
        self.gbest = self.pbest[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)
        
        # Arrays to store history for plotting
        self.avg_history = []
        self.best_history = []
        
    def optimize(self):
        for i in range(self.max_iter):
            # Generate random coefficients for stochastic behavior
            r1, r2 = np.random.rand(2)
            
            # Update velocities using PSO velocity equation:
            # v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
            self.velocities = (self.w * self.velocities + 
                             self.c1 * r1 * (self.pbest - self.positions) +
                             self.c2 * r2 * (self.gbest - self.positions))
            
            # Update positions by adding velocities
            self.positions += self.velocities
            
            # Ensure particles stay within bounds
            self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate fitness at new positions
            current_val = np.array([self.objective_func(p) for p in self.positions])
            
            # Update personal bests if current position is better
            better_positions = current_val < self.pbest_val
            self.pbest[better_positions] = self.positions[better_positions]
            self.pbest_val[better_positions] = current_val[better_positions]
            
            # Update global best if new best found
            min_val = np.min(self.pbest_val)
            if min_val < self.gbest_val:
                self.gbest = self.pbest[np.argmin(self.pbest_val)]
                self.gbest_val = min_val
            
            # Store history for plotting
            self.avg_history.append(np.mean(current_val))
            self.best_history.append(self.gbest_val)
            
        return self.gbest, self.gbest_val

# Rosenbrock function definition (banana-shaped valley)
# Global minimum at (1,1) with f(1,1)=0
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2

# Griewank function definition
# Global minimum at (0,0) with f(0,0)=0
def griewank(x):
    return 1 + (x[0]**2/400) + (x[1]**2/4000) - np.cos(x[0])*np.cos(x[1]/np.sqrt(2))

# Define test functions with their bounds and names
functions = [
    (rosenbrock, np.array([[-2, 2], [-1, 3]]), "Rosenbrock"),
    (griewank, np.array([[-30, 30], [-30, 30]]), "Griewank")
]

# Test each function
for func, bounds, name in functions:
    print(f"\nOptimizing {name} function:")
    
    # Create PSO instance and run optimization
    pso = PSO(func, bounds)
    best_position, best_value = pso.optimize()
    
    # Print results
    print(f"Best position: {best_position}")
    print(f"Best value: {best_value}")
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Plot average-so-far
    plt.subplot(1, 2, 1)
    plt.plot(pso.avg_history, 'b-')
    plt.title(f'{name} - Average-so-far')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.grid(True)
    
    # Plot best-so-far
    plt.subplot(1, 2, 2)
    plt.plot(pso.best_history, 'r-')
    plt.title(f'{name} - Best-so-far')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()