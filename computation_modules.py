import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy.sparse import csr_matrix, vstack, hstack
from tqdm import tqdm
import os
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Each function saves the calculation results in the specified directory in .npy format.
# Import evtk for VTK output
# Import vtk library
try:
    import vtk
    from vtk.util import numpy_support
    vtk_available = True
except ImportError:
    print("Warning: 'vtk' library not found. VTK file output will be disabled.")
    print("You can install it with: conda install -c conda-forge vtk")
    vtk_available = False


def fdm_calc(output_dir, M, D, Q, R, time_order, dt, T, visualization_steps):
    """
    A function that solves the reaction-diffusion equation using a second-order central difference method (FDM) on a CPU and saves the results.
    """
    print("Running FDM...")
    # --- Initial condition setup ---
    phi = np.zeros((M, 1))
    ddx = 1 / (M + 1)
    for i in range(M):
        phi[i, 0] = 0.5 - 0.5 * np.cos(2 * np.pi * (i + 1) * ddx)

    # Save initial condition phi_in (for plotting)
    phi_in = np.zeros((M + 2, 1))
    phi_in[1:M+1] = phi
    data_path = os.path.join(output_dir, "data")
    np.save(os.path.join(data_path, f'fdm_phi_initial_M{M}.npy'), phi_in)

    # --- Preparation for calculation ---
    dx = 1 / M
    phi_new = np.zeros((M, 1))

    # Construct the discrete Laplacian matrix (Dirichlet boundary conditions)
    D_h = np.zeros((M, M))
    D_h[0, 0] = -2; D_h[0, 1] = 1
    D_h[-1, -2] = 1; D_h[-1, -1] = -2
    for i in range(1, M - 1):
        D_h[i, i - 1] = 1
        D_h[i, i] = -2
        D_h[i, i + 1] = 1

    # --- Time evolution loop ---
    for nstep in tqdm(range(T + 1), mininterval=30, desc="FDM Sim"):
        # Compute diffusion and reaction terms
        phi_new = phi + (D * dt / (dx ** 2)) * (D_h @ phi) + dt * (Q * phi + R * phi ** 2)
        phi = phi_new

        # Save results at specified steps
        if nstep in visualization_steps:
            phi_save = np.zeros((M + 2, 1))
            phi_save[1:M+1] = phi
            time_val = round(nstep * dt, 4)
            np.save(os.path.join(data_path, f'fdm_phi_M{M}_t{time_val}.npy'), phi_save)

def carleman_linearization(output_dir, M, D, Q, R, time_order, dt, T, visualization_steps, trunc_order):
    """
    A function that solves nonlinear equations using the Carleman linearization (CL) method on a GPU and saves the results.
    """
    print(f"Running Carleman Linearization (K={trunc_order})...")
    # --- Preparation for calculation (GPU) ---
    dx = 1 / M
    I = cupyx.scipy.sparse.eye(M, format='csr', dtype=cp.float64)

    # --- Create initial value vector (carleman_statevector) ---
    ddx = 1 / (M + 1)
    U_dense = 0.5 - 0.5 * cp.cos(2 * cp.pi * (cp.arange(1, M + 1)) * ddx)
    U = csr_matrix(U_dense.reshape(M, 1))

    # Construct carleman_statevector (tensor product vector)
    carleman_statevector_list = [U]
    current_term = U
    for _ in range(1, trunc_order):
        current_term = cupyx.scipy.sparse.kron(U, current_term, format='csr')
        carleman_statevector_list.append(current_term)
    carleman_statevector = cupyx.scipy.sparse.vstack(carleman_statevector_list, format='csr')

    # Save initial condition carleman_statevector_in (for plotting)
    carleman_statevector_in = cp.zeros((M + 2, 1))
    carleman_statevector_in[1:M+1] = U.toarray()
    data_path = os.path.join(output_dir, "data")
    cp.save(os.path.join(data_path, f'cl_carleman_statevector_initial_M{M}_K{trunc_order}.npy'), carleman_statevector_in)

    # --- Construct the system matrix A_carle ---
    # Construct the discrete Laplacian matrix D_h (sparse matrix format)
    diag = cp.ones(M) * -2
    off_diag = cp.ones(M - 1)
    D_h = cupyx.scipy.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(M, M), format='csr', dtype=cp.float64) / (dx ** 2)

    # Linear term F_1 and nonlinear term F_2
    F_1 = D * D_h + Q * I
    
    # Convert Python scalar R to CuPy scalar before passing it
    R_cp = cp.asarray(R, dtype=cp.float64)
    F_2_data = cp.ones(M, dtype=cp.float64) * R_cp
    
    F_2_row = cp.arange(M, dtype=cp.int32)
    F_2_col = cp.arange(M) + cp.arange(M) * M
    F_2 = csr_matrix((F_2_data, (F_2_row, F_2_col)), shape=(M, M * M))

    # Generate block matrices for A_carle
    A_diag_blocks = []
    A_offdiag_blocks = []

    for n in range(1, trunc_order + 1):
        # Diagonal block A_nn
        terms_nn = []
        for k in range(n):
            term = F_1
            for _ in range(k): term = cupyx.scipy.sparse.kron(I, term, format='csr')
            for _ in range(n - k - 1): term = cupyx.scipy.sparse.kron(term, I, format='csr')
            terms_nn.append(term)
        if terms_nn:
            zero_matrix_nn = csr_matrix(terms_nn[0].shape, dtype=cp.float64)
            A_diag_blocks.append(sum(terms_nn, start=zero_matrix_nn))

        # Off-diagonal block A_n,n+1
        if n < trunc_order:
            terms_n_np1 = []
            for k in range(n):
                term = F_2
                for _ in range(k): term = cupyx.scipy.sparse.kron(I, term, format='csr')
                for _ in range(n - k - 1): term = cupyx.scipy.sparse.kron(term, I, format='csr')
                terms_n_np1.append(term)
            if terms_n_np1:
                zero_matrix_offdiag = csr_matrix(terms_n_np1[0].shape, dtype=cp.float64)
                A_offdiag_blocks.append(sum(terms_n_np1, start=zero_matrix_offdiag))

    # Assemble A_carle
    rows = []
    M_sizes = [M**n for n in range(1, trunc_order + 1)]
    for i in range(trunc_order):
        row_blocks = [csr_matrix((M_sizes[i], M_sizes[j]), dtype=cp.float64) for j in range(trunc_order)]
        row_blocks[i] = A_diag_blocks[i]
        if i + 1 < trunc_order:
            row_blocks[i+1] = A_offdiag_blocks[i]
        rows.append(hstack(row_blocks, format='csr'))
    A_carle = vstack(rows, format='csr')

    # --- Time evolution loop ---
    A_carle_dt = A_carle * dt
    for nstep in tqdm(range(T + 1), mininterval=100, desc=f"Carleman Sim (K={trunc_order})"):
        carleman_statevector = carleman_statevector + A_carle_dt @ carleman_statevector

        # Save results at specified steps
        if nstep in visualization_steps:
            carleman_statevector_save = cp.zeros((M + 2, 1))
            carleman_statevector_save[1:M+1] = carleman_statevector.toarray()[:M]
            time_val = round(nstep * dt, 4)
            cp.save(os.path.join(data_path, f'cl_carleman_statevector_M{M}_K{trunc_order}_t{time_val}.npy'), carleman_statevector_save)

def cls_calc(output_dir, M, N, D, Q, R, time_order, dt, T, visualization_steps, trunc_order, p_L, p_R):
    """
    A function that solves nonlinear equations using the Carleman linearization + Schrödingerization (CLS) method on a GPU and saves the results.
    """
    print(f"Running CLS Method (K={trunc_order}, N={N})...")
    # --- Preparation for calculation (GPU) ---
    dx = 1 / M
    dp = (p_R - p_L) / N

    # Convert Python scalar to CuPy scalar
    p_L_cp = cp.asarray(p_L, dtype=cp.float64)
    p_R_cp = cp.asarray(p_R, dtype=cp.float64)
    
    # Execute reshape to make p a two-dimensional array (1, N).
    p_1d = cp.linspace(p_L_cp, p_R_cp - dp, N)
    p = p_1d.reshape(1, N)
    
    I = cupyx.scipy.sparse.eye(M, format='csr', dtype=cp.float64)

    # --- Create initial value vector (carleman_statevector) ---
    ddx = 1 / (M + 1)
    U_dense = 0.5 - 0.5 * cp.cos(2 * cp.pi * (cp.arange(1, M + 1)) * ddx)
    U = csr_matrix(U_dense.reshape(M, 1))

    carleman_statevector_list = [U]
    current_term = U
    for _ in range(1, trunc_order):
        current_term = cupyx.scipy.sparse.kron(U, current_term, format='csr')
        carleman_statevector_list.append(current_term)
    carleman_statevector = cupyx.scipy.sparse.vstack(carleman_statevector_list, format='csr')

    # Save initial condition carleman_statevector_in (for plotting)
    carleman_statevector_in = cp.zeros((M + 2, 1))
    carleman_statevector_in[1:M+1] = U.toarray()
    data_path = os.path.join(output_dir, "data")
    cp.save(os.path.join(data_path, f'cls_carleman_statevector_initial_M{M}_K{trunc_order}_N{N}.npy'), carleman_statevector_in)

    # --- Construct the system matrix A_carle ---
    diag = cp.ones(M) * -2
    off_diag = cp.ones(M - 1)
    D_h = cupyx.scipy.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(M, M), format='csr', dtype=cp.float64) / (dx ** 2)
    F_1 = D * D_h + Q * I
    R_cp = cp.asarray(R, dtype=cp.float64)
    F_2_data = cp.ones(M, dtype=cp.float64) * R_cp
    F_2_row = cp.arange(M, dtype=cp.int32)
    F_2_col = cp.arange(M) + cp.arange(M) * M
    F_2 = csr_matrix((F_2_data, (F_2_row, F_2_col)), shape=(M, M * M))
    
    A_diag_blocks, A_offdiag_blocks = [], []
    for n in range(1, trunc_order + 1):
        terms_nn = []
        for k in range(n):
            term = F_1
            for _ in range(k): term = cupyx.scipy.sparse.kron(I, term, format='csr')
            for _ in range(n - k - 1): term = cupyx.scipy.sparse.kron(term, I, format='csr')
            terms_nn.append(term)
        if terms_nn:
            A_diag_blocks.append(sum(terms_nn, start=csr_matrix(terms_nn[0].shape, dtype=cp.float64)))
        if n < trunc_order:
            terms_n_np1 = []
            for k in range(n):
                term = F_2
                for _ in range(k): term = cupyx.scipy.sparse.kron(I, term, format='csr')
                for _ in range(n - k - 1): term = cupyx.scipy.sparse.kron(term, I, format='csr')
                terms_n_np1.append(term)
            if terms_n_np1:
                A_offdiag_blocks.append(sum(terms_n_np1, start=csr_matrix(terms_n_np1[0].shape, dtype=cp.float64)))

    rows = []
    M_sizes = [M**n for n in range(1, trunc_order + 1)]
    for i in range(trunc_order):
        row_blocks = [csr_matrix((M_sizes[i], M_sizes[j]), dtype=cp.float64) for j in range(trunc_order)]
        row_blocks[i] = A_diag_blocks[i]
        if i + 1 < trunc_order:
            row_blocks[i+1] = A_offdiag_blocks[i]
        rows.append(hstack(row_blocks, format='csr'))
    A_carle = vstack(rows, format='csr')

    # --- Construct matrices for CLS ---
    H_1 = (A_carle + A_carle.T) / 2 * dt / dp / 2
    H_2 = (A_carle - A_carle.T) / 2 * dt
    MP = sum(M_sizes)

    # --- Create initial vector W ---
    carleman_statevector_array = carleman_statevector.toarray()
    row, col, data = [], [], []
    # Calculate W and add only non-zero elements to the list
    for i in range(N):
        if p[0,i]>0:
            for j in range(MP):
                value = cp.exp(-p[0, i]) * carleman_statevector_array[j, 0]
                row.append(i * MP + j)
                col.append(0)
                data.append(value)
        else:
            for j in range(MP):
                value = cp.exp(-abs(p[0, i])) * carleman_statevector_array[j, 0]
                row.append(i * MP + j)
                col.append(0)
                data.append(value)
    # Convert to Cupy arrays
    row = cp.array(row, dtype=cp.int32)
    col = cp.array(col, dtype=cp.int32)
    data = cp.array(data, dtype=cp.float64)
    # Create W as a sparse matrix in CSR format
    W = csr_matrix((data, (row, col)), shape=(MP * N, 1))

    # --- Compute B matrix ---
    B_1 = cp.eye((N), dtype=cp.float64)
    B_1 = csr_matrix(B_1)
    I_B = cp.eye((MP), dtype=cp.float64)
    # Create sparse matrix B_2
    row, col, data = [], [], []
    for i in range(N - 1):
        row.append(i)
        col.append(i + 1)
        data.append(1.0)
    # Exception handling for the last line
    row.append(N - 1)
    col.append(0)
    data.append(1.0)
    # Convert the list to a cupy array and specify the data type.
    row = cp.array(row, dtype=cp.int32)
    col = cp.array(col, dtype=cp.int32)
    data = cp.array(data, dtype=cp.float64)
    # Create B_2 in CSR format.
    B_2 = csr_matrix((data, (row, col)), shape=(N, N))
    # Create sparse matrix B_3
    row, col, data = [], [], []
    # Exception handling for the first row
    row.append(0)
    col.append(N-1)
    data.append(1.0)
    for i in range(1,N):
        row.append(i)
        col.append(i - 1)
        data.append(1.0)
    # Convert the list to a cupy array and specify the data type.
    row = cp.array(row, dtype=cp.int32)
    col = cp.array(col, dtype=cp.int32)
    data = cp.array(data, dtype=cp.float64)
    # Create B_2 in CSR format.
    B_3 = csr_matrix((data, (row, col)), shape=(N, N))
    B_diag=I_B+H_2
    B_diag=csr_matrix(B_diag)
    # Compute the Kronecker product
    B = cupyx.scipy.sparse.kron(B_1,B_diag)+cupyx.scipy.sparse.kron(B_2,-H_1)+cupyx.scipy.sparse.kron(B_3,H_1)
    # Creating a directory for saving VTK files
    vtk_dir = os.path.join(output_dir, "vtk_files_vtk")
    if vtk_available and not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)

    # --- Time evolution loop ---
    for nstep in tqdm(range(T + 1), mininterval=100, desc=f"CLS Sim (K={trunc_order}, N={N})"):
        W = B @ W

        # Save results at specified steps
        if nstep in visualization_steps:
            W_array = W.toarray()
            time_val = round(nstep * dt, 4)

            # 3D plot data saving (wp_visual)
            wp_visual = cp.zeros((N, M + 2))
            for i in range(N):
                exp_p = cp.exp(abs(p[0, i]))
                # wp_visual[i, 1:M+1] = exp_p * W_array[i*MP : i*MP+M].flatten()
                wp_visual[i, 1:M+1] = W_array[i*MP : i*MP+M].flatten()
            cp.save(os.path.join(data_path, f'cls_visual_M{M}_K{trunc_order}_N{N}_t{time_val}.npy'), wp_visual)

            # Save phi(x) at a specific position in p-space (wp_phi)
            p_id = N // 2 + 3
            wp_save = cp.zeros((M + 2, 1))
            wp_save[1:M+1] = cp.exp(p[0, p_id]) * W_array[p_id*MP : p_id*MP+M]
            cp.save(os.path.join(data_path, f'cls_phi_x_M{M}_K{trunc_order}_N{N}_t{time_val}.npy'), wp_save)

            # Save phi(p) at a specific position in x-space (wp_p)
            x_id = M // 2
            wp_p_save = W_array[x_id::MP].flatten()
            cp.save(os.path.join(data_path, f'cls_phi_p_M{M}_K{trunc_order}_N{N}_t{time_val}.npy'), wp_p_save)

            # --- VTK file output processing ---
            if vtk_available:
                # Preparing the coordinate axes (Numpy array)
                x_coords = np.linspace(0, 1, M + 2)
                p_coords = p.get().flatten()
                z_coords = np.array([0.0])

                # Create VTK structured grid object
                structuredGrid = vtk.vtkStructuredGrid()
                structuredGrid.SetDimensions(len(x_coords), len(p_coords), len(z_coords))

                # Set the coordinates of the grid points
                points = vtk.vtkPoints()
                for z in z_coords:
                    for y in p_coords:
                        for x in x_coords:
                            points.InsertNextPoint(x, y, z)
                structuredGrid.SetPoints(points)

                # Prepare the data body as a Numpy array
                phi_data_np = wp_visual.get().T # (M+2, N) の形状
                # Convert to a 1D array for VTK (Fortran order 'F' is important)
                phi_flat = phi_data_np.flatten(order='F')

                # Convert to VTK array and set the name
                vtk_phi_array = numpy_support.numpy_to_vtk(phi_flat, deep=True, array_type=vtk.VTK_DOUBLE)
                vtk_phi_array.SetName("phi")

                # Set as point data
                structuredGrid.GetPointData().SetScalars(vtk_phi_array)

                # Prepare the writer and write to file
                writer = vtk.vtkXMLStructuredGridWriter()
                vtk_filename = os.path.join(vtk_dir, f"phi_data_t_{time_val:.4f}.vts")
                writer.SetFileName(vtk_filename)
                writer.SetInputData(structuredGrid)
                writer.Write()