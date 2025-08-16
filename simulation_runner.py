import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import requests
import cupy as cp

# import a custom module
import computation_modules as cm

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- 1. Initialization and Parameter Loading ---

    # GPU device settings
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # CuPy memory pool settings
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Move to the project root directory
    # Note: Change this path according to your environment
    # Load configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    path = config['path']
    try:
        os.chdir(path)
    except FileNotFoundError:
        print("Warning: Could not change directory to your project root. Using current directory.")

    print(f"Current Directory: {os.getcwd()}")

    # Expand parameters into variables
    params = config['simulation_parameters']
    time_params = config['time_parameters']
    p_params = config['p_space_parameters']
    
    M_values = params['M_values']
    N_values = params['N_values']
    trunc_order_values = params['trunc_order_values']
    D, Q, R = params['D'], params['Q'], params['R']
    p_L, p_R = p_params['p_L'], p_params['p_R']
    
    time_order = time_params['time_order']
    dt = 1.0 * 10**(-time_order)
    T = int(time_params['total_time_factor'] * 10**time_order)

    # Calculate the time steps to save data
    num_viz_points = time_params['visualization_points']
    visualization_steps = [int((T / num_viz_points) * (i + 1)) for i in range(num_viz_points)]
    # Add the initial time step (t=0) to the beginning of the list
    visualization_steps.insert(0, 0)

    # --- 2. Create Results Directory ---

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = 'jax_simulation' # File name prefix
    output_dir = os.path.join("results", f"{current_time_str}_{file_name}")
    data_dir = os.path.join(output_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # Save the runtime settings to the results directory (for reproducibility)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- 3. Run Simulation ---
    start_time = time.time()

    try:
        # Parameter loop
        for m in M_values:
            # --- Finite Difference Method (2nd order central difference method) ---
            cm.fdm_calc(
                output_dir=output_dir, M=m, D=D, Q=Q, R=R,
                time_order=time_order, dt=dt, T=T,
                visualization_steps=visualization_steps
            )

            for k in trunc_order_values:
                # --- Carleman Linearization ---
                cm.carleman_linearization(
                    output_dir=output_dir, M=m, D=D, Q=Q, R=R,
                    time_order=time_order, dt=dt, T=T,
                    visualization_steps=visualization_steps, trunc_order=k
                )
                
                for n in N_values:
                    # --- CLS Method ---
                    cm.cls_calc(
                        output_dir=output_dir, M=m, N=n, D=D, Q=Q, R=R,
                        time_order=time_order, dt=dt, T=T,
                        visualization_steps=visualization_steps, trunc_order=k,
                        p_L=p_L, p_R=p_R
                    )

    except Exception as e:
        error_message = f"An error occurred during simulation: {e}"
        print(error_message)

    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_message = f"Simulation run '{file_name}' finished. Total time: {elapsed_time:.2f} seconds."
        print(final_message)