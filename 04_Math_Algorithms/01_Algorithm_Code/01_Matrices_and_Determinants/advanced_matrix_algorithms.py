"""
=============================================================================
Advanced Linear Algebra & Machine Learning Core Engine
=============================================================================

A pure Python, zero-dependency mathematical engine built from first principles.
This module provides production-grade algorithms for advanced matrix transformations,
dimensionality reduction, and feature extraction without relying on external 
C-optimized libraries (like NumPy or SciPy).

It is designed with an object-oriented architecture, utilizing custom `Matrix` 
objects to handle both square (N x N) and rectangular (N x M) multi-dimensional 
datasets.

Core Algorithms & Components:
-----------------------------
1. Determinant (get_determinant):
    Calculates the scaling factor of the linear transformation, crucial for understanding
    matrix invertibility.

2. Matrix Inverse (get_inverse):
    Computes the multiplicative inverse of a square matrix using foundational row 
    operations/minors.

3. Linear System Solver (`solve_linear_system`):
   Computes the solution vector 'x' for systems of linear equations (Ax = b) 
   by leveraging the foundational matrix inverse engine.

4. Normal Pseudo-Inverse (`get_pseudo_inverse`):
   Calculates the exact Moore-Penrose Pseudo-Inverse for rectangular matrices 
   using the Left Inverse formula, essential for Least Squares regression.
    
5. Matrix Trace (`get_matrix_trace`):
   Computes the invariant sum of main diagonal elements, mathematically linked 
   to the sum of all eigenvalues.

6. Eigen-Decomposition (`get_dominant_eigenvector`):
   Utilizes the Power Iteration algorithm to iteratively extract the dominant 
   eigenvector and eigenvalue, revealing the strongest variance in square matrices.
   
7. Row Echelon Form (`get_row_echelon_matrix`):
   Transforms matrices into Row Echelon Form via Gaussian Elimination to compute 
   the true mathematical rank and analyze spatial dimensions.
   
8. ML Feature Extraction (`get_independent_features_matrix`):
   Applies an index-tracking elimination algorithm to safely discard linearly 
   dependent (duplicate) dimensions while preserving pure original data for AI training.
   
9. Singular Value Decomposition - SVD (`get_dominant_svd`):
   The core engine for Dimensionality Reduction. Decomposes any rectangular dataset 
   to extract its dominant singular values and orthogonal basis vectors (U, Sigma, V), 
   forming the foundational backend for PCA, compression, and recommendation systems.

10. SVD Pseudo-Inverse (`get_pseudo_inverse_svd`):
   Computes a robust Rank-1 Approximated Pseudo-Inverse utilizing Singular 
   Value Decomposition to handle noise and ill-conditioned data safely.

Architecture Note:
------------------
All functions in this module interface seamlessly with the custom `Matrix` class 
from `matrix_operation.py`, ensuring deep modularity, memory safety, and 
algorithmic optimization (e.g., O(N^2) mathematical bypasses in the SVD pipeline).
"""

from matrix_operation import Matrix
import math
import random

def get_determinant(matrix_obj):
    """
    Calculates the determinant of a square matrix.

    Logic:
        1. Validation: Checks if the matrix is square (rows == columns).
        2. 1x1 Edge Case: Returns the single element directly.
        3. Delegation: Passes the raw 2D list to the private recursive engine 
           ('_recursive_det') for N x N calculation.

    Args:
        matrix_obj (Matrix): The custom Matrix object.

    Returns:
        int/float: The final determinant value.
        
    Raises:
        ValueError: If the input matrix is not a square matrix.
    """
    
    # Validation for Square Matrix
    if matrix_obj.row != matrix_obj.col:
        raise ValueError("Determinant only exists for square matrices.")
    
    # 1x1 Edge Case
    if matrix_obj.col == 1 and matrix_obj.row == 1:
        return matrix_obj.matrix[0][0]

    # Pass the internal raw grid (.matrix) to the recursive helper
    return _recursive_det(matrix_obj.matrix)

def _recursive_det(matrix_grid):
    """
    The core Private Recursive Engine to calculate the determinant.

    Step-by-Step Logic Breakdown:
        1. Base Case: If the matrix shrinks to 2x2, it applies the direct formula: (a11 * a22) - (a12 * a21).
        2. Expansion Loop: Iterates through each element of the FIRST row (index j).
        3. Sign Alternation: Calculates the mathematical sign (+, -, +, -) using (-1)**j.
        4. Sub-Matrix Creation: Creates a minor by skipping the first row and removing the current column 'j'.
        5. Recursive Call: Multiplies the sign, element, and the recursive determinant of the sub-matrix.
    
    Args:
        matrix_grid (list): A 2D Python list representing the matrix grid.

    Returns:
        int/float: The calculated determinant.
    """
    # Base Case: 2x2 Matrix
    if len(matrix_grid) == 2:
        return matrix_grid[0][0] * matrix_grid[1][1] - matrix_grid[0][1] * matrix_grid[1][0]
    
    ans = 0
    # Expansion Loop along the first row
    for j in range(len(matrix_grid)):
        element = matrix_grid[0][j]
        sign = (-1)**j

        # Sub-Matrix Creation (Minor)
        sub_matrix = []
        for row in matrix_grid[1:]:
            cut_row = row[:j] + row[j+1:]
            sub_matrix.append(cut_row)

        # Recursive Call
        ans += sign * element * _recursive_det(sub_matrix)

    return ans


def get_inverse(matrix_obj):
    """
    Computes the multiplicative inverse of a square matrix.

    This acts as a wrapper/manager function. It strictly validates the mathematical 
    possibility of inversion (must be square, determinant != 0), handles 1x1 and 2x2 
    edge cases in O(1) time for performance, and delegates highly complex N x N 
    matrices to the recursive backend engine.

    Args:
        matrix_obj (Matrix): The custom Matrix object to be inverted.

    Returns:
        Matrix: A new Matrix object containing the inverted data.
        
    Raises:
        ValueError: If the matrix is non-square or singular (determinant is 0).
    """
    
    # ---------------------------------------------------------
    # VALIDATION PHASE
    # ---------------------------------------------------------
    if matrix_obj.row != matrix_obj.col:
        raise ValueError("Inverse only exists for square (N x N) matrices.")
    
    grid = matrix_obj.matrix
    n = matrix_obj.row

    # ---------------------------------------------------------
    # O(1) EDGE CASE: 1x1 MATRIX
    # ---------------------------------------------------------
    if n == 1:
        if grid[0][0] == 0:
            raise ValueError("Singular Matrix (Determinant = 0). Inverse does not exist.")
        
        return Matrix(1, 1, data=[[1 / grid[0][0]]])
    
    # ---------------------------------------------------------
    # O(1) EDGE CASE: 2x2 MATRIX (Direct Formula)
    # ---------------------------------------------------------
    if n == 2:
        a11, a12 = grid[0][0], grid[0][1]
        a21, a22 = grid[1][0], grid[1][1]

        det = (a11 * a22) - (a12 * a21)
        if det == 0:
            raise ValueError("Singular Matrix (Determinant = 0). Inverse does not exist.")
        
        # Swapping logic for Adjoint of 2x2
        adjA = [[a22, -a12], 
                [-a21, a11]]

        # Applying (1/Det) * Adjoint
        inverse_grid = [[(element / det) for element in row] for row in adjA]
        
        return Matrix(2, 2, data=inverse_grid)

    # ---------------------------------------------------------
    # HEAVY LIFTING: N x N MATRIX DELEGATION
    # ---------------------------------------------------------
    inverted_grid = _calculate_nxn_inverse(grid)
    return Matrix(n, n, data=inverted_grid)

def _calculate_nxn_inverse(matrix_grid):
    """
    The core recursive engine implementing the 'Cofactor Expansion' method 
    (the traditional school mathematics approach) scaled up algorithmically for N x N.

    Step-by-Step Architecture:
        1. Cofactor Matrix: Generates a matrix of signs and sub-matrix determinants (minors).
        2. Smart Determinant: Reuses the first row's cofactors to compute the total det.
        3. Adjoint Matrix: Transposes the Cofactor Matrix.
        4. Final Division: Multiplies every element by (1 / total determinant).

    Args:
        matrix_grid (list): A 2D list representing the raw matrix grid.

    Returns:
        list: A 2D list representing the inverted matrix data.
    """
    n = len(matrix_grid)
    
    # ---------------------------------------------------------
    # STEP 1: COFACTOR MATRIX CALCULATION
    # ---------------------------------------------------------
    co_factor_matrix = []
    
    for r in range(n):
        co_factor_row = []
        for c in range(n):
            # Chessboard pattern for signs (+, -, +, -)
            sign = (-1)**(r + c)

            # Sub-Matrix Creation (Eliminating current row 'r' and col 'c')
            sub_matrix = []
            for row in matrix_grid[:r] + matrix_grid[r+1:]:
                cut_row = row[:c] + row[c+1:]
                sub_matrix.append(cut_row)

            # Call our internal _recursive_det for the minor
            minor_det = _recursive_det(sub_matrix)
            co_factor_row.append(sign * minor_det)
            
        co_factor_matrix.append(co_factor_row)
        
    # ---------------------------------------------------------
    # STEP 2: DETERMINANT CALCULATION
    # ---------------------------------------------------------
    # Instead of recalculating the full determinant, we multiply 
    # the original first row with the calculated first row of cofactors.
    det = sum(matrix_grid[0][i] * co_factor_matrix[0][i] for i in range(n))
    
    if det == 0:
        raise ValueError("Singular Matrix (Determinant = 0). Inverse does not exist.")

    # ---------------------------------------------------------
    # STEP 3: ADJOINT MATRIX (Transpose of Cofactors)
    # ---------------------------------------------------------
    adjA = []
    for c in range(n):
        new_row = []
        for r in range(n):
            new_row.append(co_factor_matrix[r][c])
        adjA.append(new_row)

    # ---------------------------------------------------------
    # STEP 4: FINAL INVERSE COMPUTATION
    # ---------------------------------------------------------
    # Formula: A^-1 = (1 / Det(A)) * Adj(A)
    inverse_grid = [[(element / det) for element in row] for row in adjA]

    return inverse_grid


def solve_linear_system(A_matrix, b_matrix):
    """
    Solves a system of linear equations of the form Ax = b.
    Mathematically computes: x = A^-1 * b

    Args:
        A_matrix (Matrix): The coefficient matrix (must be square).
        b_matrix (Matrix): The constant vector matrix (N x 1).

    Returns:
        Matrix: The solution vector 'x'.
    """
    # 1. Calculate Inverse of A
    A_inv = get_inverse(A_matrix)
    
    # 2. Multiply Inverse of A with b (Using your multiply engine)
    solution_matrix = A_inv.multiply(b_matrix)
    
    return solution_matrix


def get_pseudo_inverse(matrix_obj):
    """
    Calculates the Moore-Penrose Pseudo-Inverse for Rectangular Matrices (N x M).
    Formula used (Left Inverse for full column rank): (A^T * A)^-1 * A^T
    
    This is heavily used in Machine Learning (Linear Regression / Least Squares)
    when the dataset is not a square matrix.

    Args:
        matrix_obj (Matrix): The rectangular matrix.

    Returns:
        Matrix: The pseudo-inverse matrix.
    """
    # Step 1: A^T (Transpose)
    A_T = matrix_obj.transpose()
    
    # Step 2: (A^T * A)
    A_T_A = A_T.multiply(matrix_obj)
    
    # Step 3: (A^T * A)^-1
    inv_A_T_A = get_inverse(A_T_A)
    
    # Step 4: Multiply by A^T
    pseudo_inv = inv_A_T_A.multiply(A_T)
    
    return pseudo_inv


def get_matrix_trace(matrix_obj):
    """
    Computes the Trace of a matrix (the sum of its main diagonal elements).

    The trace is a fundamental invariant in linear algebra. It remains constant 
    under basis transformations and mathematically equals the sum of all eigenvalues 
    of the matrix. This function is dynamically designed to handle both square (N x N) 
    and rectangular (N x M) matrices without throwing index errors.

    Args:
        matrix_obj (Matrix): The custom Matrix object whose trace is to be calculated.

    Returns:
        float/int: The numerical sum of the main diagonal elements.
    """
    
    # ---------------------------------------------------------
    # STEP 1: INITIALIZATION
    # ---------------------------------------------------------
    diag_sum = 0
    i = 0
    
    # ---------------------------------------------------------
    # STEP 2: DIAGONAL TRAVERSAL (The Logic)
    # ---------------------------------------------------------
    # We iterate diagonally (where row_index == col_index).
    # The condition 'i < row' AND 'i < col' elegantly ensures that the loop 
    # stops automatically before going out of bounds, making it 100% safe 
    # for rectangular matrices (e.g., 10000 x 50).
    while i < matrix_obj.row and i < matrix_obj.col:
        
        # Extract the diagonal element and add it to our accumulator
        diag_sum += matrix_obj.matrix[i][i]
        
        # Move one step diagonally down-right
        i += 1
        
    return diag_sum


def get_dominant_eigenvector(matrix, iterations=100):
    """
    Calculates the dominant eigenvector and eigenvalue using the Power Iteration method.

    This algorithm repeatedly applies a matrix transformation to a random starting
    vector. Over time, the vector naturally aligns with the matrix's most powerful 
    transformation direction (the Dominant Eigenvector). To prevent the vector's 
    values from exploding (floating-point overflow), it normalizes the vector using 
    the Infinity Norm (maximum absolute value) after every multiplication.

    Args:
        matrix (Matrix): A custom Matrix object representing an N x N transformation.
                         It must have a 'col' attribute, a 'matrix' data grid, and 
                         support Matrix.multiply() and Matrix.scalar_divide().
        iterations (int, optional): The number of times to apply the transformation loop. 
                                    Higher iterations yield more precise results. 
                                    Defaults to 100.

    Returns:
        None: The function directly prints the Dominant Eigenvector and Eigenvalue.
    """
    
    # STEP 1: Generate a random "dummy" vector (The starting leaf in the tornado)
    # We reshape it to be a pure column vector (N rows, 1 column)
    random_vector = [random.random() for _ in range(matrix.col)]
    v_new = Matrix(matrix.col, 1, data=random_vector)
    
    # STEP 2: The Power Iteration Loop
    for reps in range(iterations):
        
        # Apply the Matrix Action (Stretch and rotate the vector in space)
        v_new = Matrix.multiply(matrix, v_new)
        
        # Find the maximum absolute value in the current vector (Infinity Norm Hack)
        # This identifies the largest component to scale the vector down.
        max_val = max(abs(row[0]) for row in v_new.matrix)
        
        # Normalize the vector by dividing all elements by the max_val.
        # This keeps the vector length stable. By the end of the loop, 
        # this scaling factor (max_val) perfectly converges into the Eigenvalue!
        v_new = Matrix.scalar_divide(v_new, max_val)

    # STEP 3: Output the extracted mathematical properties
    # Return the exact Matrix object and the float scalar
    return v_new, max_val


def get_row_echelon_matrix(input_matrix):
    """
    Computes the Matrix Rank by mathematically transforming the dataset into its 
    Row Echelon Form using the Gaussian Elimination algorithm.

    This function systematically eliminates linearly dependent (duplicate) dimensions. 
    Note for Mathematicians: The surviving rows in the returned matrix will have 
    mathematically altered values (changed magnitudes and basis) compared to the 
    original data. However, geometrically, these transformed vectors span the EXACT 
    same vector space and point in the same foundational directions as the original. 
    
    This is a pure mathematical engine designed to reveal the core transformation 
    space of the matrix, rather than preserving raw data for Machine Learning.

    Args:
        input_matrix (Matrix): The original custom Matrix object to be evaluated.

    Returns:
        tuple: A tuple containing two elements:
            - rank (int): The true mathematical rank (number of independent dimensions).
            - echelon_matrix (Matrix): A new Matrix object containing the mathematically 
                                       reduced (Row Echelon) data grid.
    """
    
    # STEP 1: Extract the raw grid from the custom Matrix object
    # We create a deep copy so we don't destroy the original data in memory
    mat = [row[:] for row in input_matrix.matrix]
    rows = len(mat)
    cols = input_matrix.col  # Extracting column property directly from your object
    
    rank = 0
    
    # STEP 2: The Duplicate Killer Algorithm (Gaussian Elimination)
    for c in range(cols):
        pivot_row = rank
        
        # Find a valid pivot (non-zero number)
        while pivot_row < rows and abs(mat[pivot_row][c]) < 1e-9:
            pivot_row += 1
            
        if pivot_row == rows:
            continue  # Empty dimension, skip
            
        # Swap the valid row to the top of our working area
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        
        # Eliminate all rows below the current pivot
        for i in range(rank + 1, rows):
            if abs(mat[i][c]) > 1e-9:
                factor = mat[i][c] / mat[rank][c]
                for j in range(c, cols):
                    mat[i][j] -= factor * mat[rank][j]
                    
        rank += 1
        
    # STEP 3: Extract the surviving (independent) rows
    filtered_grid = []
    for row in mat:
        # If the row has any number that is not mathematically zero, it survives
        if any(abs(val) > 1e-9 for val in row):
            filtered_grid.append(row)
            
    # STEP 4: THE MOVE (Reconstructing the Matrix Object)
    # The columns remain exactly the same as the original matrix.
    # The new rows count is simply the length of our surviving grid.
    new_rows = len(filtered_grid)
    new_cols = cols
    
    # Package everything back into your custom Matrix engine!
    filtered_matrix_obj = Matrix(new_rows, new_cols, data=filtered_grid)
    
    return rank, filtered_matrix_obj


def get_independent_features_matrix(input_matrix):
    """
    Extracts the true, linearly independent rows (features) from a dataset, 
    preserving their original mathematical magnitude and values.

    Unlike the mathematical 'Row Echelon Form' which modifies the raw data to 
    find the rank, this Machine Learning pipeline uses 'Index Tracking' alongside 
    Gaussian Elimination. It identifies dependent (duplicate) dimensions and safely 
    extracts only the original, untouched data required for training AI models.

    Args:
        input_matrix (Matrix): A custom Matrix object containing the raw, potentially 
                               redundant dataset.

    Returns:
        tuple: A tuple containing two elements:
            - rank (int): The true mathematical dimension (number of unique features).
            - clean_matrix (Matrix): A new custom Matrix object containing ONLY the 
                                     original independent rows.
    """
    
    # ---------------------------------------------------------
    # STEP 1: INITIALIZATION & SAFETY DEEP COPY
    # ---------------------------------------------------------
    # We copy the raw data to perform our mathematical 'operation' without 
    # corrupting the original dataset in the memory.
    mat = [row[:] for row in input_matrix.matrix]
    rows = len(mat)
    cols = input_matrix.col
    
    # THE LOGIC: INDEX TRACKER
    # We create a list of indices [0, 1, 2, ..., N] representing the original rows.
    # As Gaussian Elimination swaps rows to destroy duplicates, we will swap these 
    # indices as well. This allows us to trace back to the original untouched data.
    original_indices = list(range(rows))
    
    rank = 0
    
    # ---------------------------------------------------------
    # STEP 2: GAUSSIAN ELIMINATION (THE DUPLICATE KILLER)
    # ---------------------------------------------------------
    for c in range(cols):
        pivot_row = rank
        
        # Search for a valid pivot element (non-zero value to act as our multiplier)
        # We use < 1e-9 to avoid floating point precision errors (treating 0.000000001 as 0)
        while pivot_row < rows and abs(mat[pivot_row][c]) < 1e-9:
            pivot_row += 1
            
        # If the entire column is zero, it yields no information. Skip to the next.
        if pivot_row == rows:
            continue
            
        # Swap the found valid row to our current 'rank' position (Top of the working block)
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        
        # CRUCIAL: Swap the trackers to maintain the mapping to the original dataset
        original_indices[rank], original_indices[pivot_row] = original_indices[pivot_row], original_indices[rank]
        
        # Eliminate all data below the current pivot row
        for i in range(rank + 1, rows):
            if abs(mat[i][c]) > 1e-9:
                # Calculate how many times the pivot row fits into the target row
                factor = mat[i][c] / mat[rank][c]
                
                # Subtract the scaled pivot row from the target row
                for j in range(c, cols):
                    mat[i][j] -= factor * mat[rank][j]
                    
        # A unique dimension was successfully isolated and processed
        rank += 1
        
    # ---------------------------------------------------------
    # STEP 3: RECONSTRUCTING THE DATASET
    # ---------------------------------------------------------
    # The first 'rank' number of items in our tracker array now hold the 
    # exact original indices of the rows that survived the elimination.
    surviving_indices = original_indices[:rank]
    
    # Sort them to maintain the sequential order of the original dataset
    surviving_indices.sort()
    
    filtered_raw_data = []
    for idx in surviving_indices:
        # Fetch the pristine, unmodified data directly from the input object
        filtered_raw_data.append(input_matrix.matrix[idx])
            
    # Package the pure data back into a scalable Matrix object for the AI engine
    clean_matrix_obj = Matrix(len(filtered_raw_data), cols, data=filtered_raw_data)
    
    return rank, clean_matrix_obj


def get_dominant_svd(matrix_obj):
    """
    Computes the Dominant Singular Value Decomposition (Rank-1 Approximation) 
    for any rectangular matrix using foundational mathematical operations.

    This function acts as a core dimensionality reduction engine. Instead of 
    relying on external C-libraries, it chains together custom Matrix objects 
    (transpose, multiply) and a Power Iteration Eigenvector engine to extract 
    the single most powerful pattern (Principal Component) from the dataset.

    Mathematical Pipeline: A = U * Sigma * V^T

    Args:
        matrix_obj (Matrix): The input data matrix. Can be of any shape (N x M),
                             representing raw, uncompressed data.

    Returns:
        tuple: A tuple containing the three dominant components of the SVD:
            - u_vector (list): The Left Singular Vector (Row/User alignments).
            - sigma (float): The Dominant Singular Value (Pattern strength/magnitude).
            - v_vector (list): The Right Singular Vector (Column/Feature alignments).
    """

    # ---------------------------------------------------------
    # STEP 1: PREPARE THE SPACES (Transpose)
    # ---------------------------------------------------------
    # Utilize the built-in N x M transpose feature of the custom Matrix class.
    # If matrix_obj is (N x M), a_transpose becomes (M x N).
    a_transpose = matrix_obj.transpose()

    # ---------------------------------------------------------
    # STEP 2: BUILD THE COVARIANCE-LIKE MATRIX (A^T * A)
    # ---------------------------------------------------------
    # Multiply the transposed matrix by the original matrix.
    # This guarantees a perfect Square Matrix (M x M) representing feature correlations,
    # which is strictly required for the Eigenvector engine to function.
    ata_matrix = a_transpose.multiply(matrix_obj)

    # ---------------------------------------------------------
    # STEP 3: EXTRACT THE FEATURE BASIS (Right Singular Vector - V)
    # ---------------------------------------------------------
    # Pass the square covariance matrix into our custom Power Iteration engine.
    # v_vector dictates how the original features should be rotated/aligned.
    v_vector, eigenvalue = get_dominant_eigenvector(ata_matrix)

    # ---------------------------------------------------------
    # STEP 4: EXTRACT THE MAGNITUDE (Singular Value - Sigma)
    # ---------------------------------------------------------
    # The singular value is the square root of the eigenvalue derived from A^T * A.
    # We use max(0.0, eigenvalue) to safeguard against floating-point precision 
    # errors yielding infinitesimally small negative numbers.
    sigma = math.sqrt(max(0.0, eigenvalue))

    # Safety Check: If Sigma is effectively zero, the matrix contains no data/variance.
    # We return zeroed vectors to prevent division by zero in the next step.
    if sigma < 1e-9:
        return [0.0] * matrix_obj.row, 0.0, v_vector

    # ---------------------------------------------------------
    # STEP 5: THE BYPASS (Left Singular Vector - U)
    # ---------------------------------------------------------
    # Instead of computing A * A^T (which creates a massive N x N matrix and crashes memory),
    # we use the geometric property of SVD: U = (A * V) / Sigma
    # This optimizes our time complexity drastically from O(N^3) to O(N^2).
    
    rows = matrix_obj.row
    cols = matrix_obj.col
    u_vector = [0.0] * rows
    
    # Perform matrix-vector multiplication: A (N x M) multiplied by V (M x 1)
    for i in range(rows):
        dot_product = 0.0
        for j in range(cols):
            # Since v_vector is a Matrix object (an N x 1 column vector),
            # we must access its data using .matrix[row][col]
            v_val = v_vector.matrix[j][0] 
            
            dot_product += matrix_obj.matrix[i][j] * v_val
            
        u_vector[i] = dot_product / sigma

    # =========================================================
    # PACKAGING RAW LISTS INTO MATRIX OBJECTS BEFORE RETURNING
    # =========================================================
    
    # 1. Convert u_vector (List of size M) into an M x 1 Column Matrix
    if isinstance(u_vector, list):
        U_matrix = Matrix(matrix_obj.row, 1, data=[[val] for val in u_vector])
    else:
        U_matrix = u_vector # If it's already a Matrix

    # 2. Convert v_vector (List of size N) into an N x 1 Column Matrix
    if isinstance(v_vector, list):
        V_matrix = Matrix(matrix_obj.col, 1, data=[[val] for val in v_vector])
    else:
        V_matrix = v_vector

    # Return strict Matrix objects and the scalar sigma
    return U_matrix, sigma, V_matrix


def get_pseudo_inverse_svd(matrix_obj):
    """
    Calculates the Rank-1 Approximated Moore-Penrose Pseudo-Inverse.
    Formula: A^+ = V * (1 / sigma) * U^T

    Args:
        matrix_obj (Matrix): The rectangular input matrix.

    Returns:
        Matrix: The approximated pseudo-inverse matrix.
    """
    # 1. Get STRICT Matrix objects from your upgraded SVD engine
    U, sigma_val, V = get_dominant_svd(matrix_obj)
    
    # 2. Transpose U to make it a Row Vector (1 x M)
    U_T = U.transpose()
    
    # 3. Calculate Sigma^+ (Inverse of the single dominant singular value)
    if sigma_val > 1e-10:
        sigma_plus_val = 1 / sigma_val
    else:
        sigma_plus_val = 0

    # 4. The Rank-1 Inverse Computation
    # First: V * U^T (Outer Product gives an N x M matrix)
    outer_product_matrix = V.multiply(U_T)
    
    # Second: Scale the entire matrix by (1 / sigma)
    pseudo_inv = outer_product_matrix.scalar_multiply(sigma_plus_val)
    
    return pseudo_inv


if __name__ == "__main__":

    # =====================================================================
    # TESTING: DETERMINANT & INVERSE ENGINE
    # =====================================================================

    print("\nCORE MATH ENGINE TESTING...\n")

    # A 3x3 Non-Singular Matrix (Determinant is not 0)
    test_data = [
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ]
    
    m_test = Matrix(3, 3, data=test_data)
    
    print("--- ORIGINAL INPUT MATRIX ---")
    print(m_test)
    print("\n" + "="*50 + "\n")

    # ---------------------------------------------------------
    # TEST 1: DETERMINANT ENGINE
    # ---------------------------------------------------------
    print("--- 1. DETERMINANT TEST ---")
    try:
        det_value = get_determinant(m_test)
        print(f"✅ Calculated Determinant: {det_value}")
    except ValueError as e:
        print(f"❌ Determinant Error: {e}")

    print("\n" + "="*50 + "\n")

    # ---------------------------------------------------------
    # TEST 2: INVERSE MATRIX ENGINE
    # ---------------------------------------------------------
    print("--- 2. INVERSE MATRIX TEST ---")
    try:
        inv_matrix_obj = get_inverse(m_test)
        print("✅ Calculated Inverse Matrix:")
        
        # FRONTEND FORMATTING: 
        # The backend data is 100% mathematically pure (unrounded floats).
        # We only format it to 2 decimal places for a clean, aligned display.
        print(inv_matrix_obj)
            
    except ValueError as e:
        print(f"❌ Inverse Error: {e}")
        
    print("\n🎯 TESTING COMPLETE.\n")

    # =====================================================================
    # TESTING: LINEAR EQUATION SOLVER (Ax = b)
    # =====================================================================
    print("TESTING LINEAR SYSTEM SOLVER (Ax = b)...\n")

    # Equation 1: 2x + 3y = 8
    # Equation 2: 5x - 1y = 3
    # Expected Mathematical Answer: x = 1, y = 2

    A_data = [
        [2, 3],
        [5, -1]
    ]
    
    b_data = [
        [8],
        [3]
    ]

    # Initialize matrices
    A = Matrix(2, 2, data=A_data)
    b = Matrix(2, 1, data=b_data)

    print("--- MATRIX A (Coefficients) ---")
    print(A)

    print("\n--- MATRIX b (Constants) ---")
    print(b)

    print("\n" + "="*40)

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------
    try:
        solution = solve_linear_system(A, b)
        
        print("\n✅ Solution Vector 'x' (Calculated):")
        
        print(solution)
            
    except Exception as e:
        print(f"❌ Error solving system: {e}")

    print("\n🎯 TESTING COMPLETE.")

    # ==========================================
    # TESTING THE TRACE ENGINE
    # ==========================================
    # Rectangular Matrix Test (3x4)
    data = [
        [10, 2, 3, 5],
        [4, 20, 6, 7],
        [7, 8, 30, 9]
    ]
    
    m1 = Matrix(3, 4, data=data)
    trace_value = get_matrix_trace(m1)
    
    print("\n=== TRACE ENGINE RESULTS ===\n")
    print(f"Matrix Dimension: {m1.row}x{m1.col}")
    # The sum should be: 10 + 20 + 30 = 60
    print(f"Calculated Trace: {trace_value}")


    # ==========================================
    # TESTING THE CUSTOM EIGENVECTOR ENGINE
    # ==========================================
    # Generating a 10x10 dynamic dataset using numbers 0 to 99
    l1 = [i for i in range(100)]
    m1 = Matrix(10, 10, data=l1)

    # Capture the returned objects in variables
    eigenvector, eigenvalue = get_dominant_eigenvector(m1)

    print("\n=== ENGINE RESULTS ===\n")
    print("Dominant Eigenvector (Matrix Object):", eigenvector, sep="\n")
    print("\nDominant Eigenvalue (Float):", eigenvalue)

    
    # ==========================================
    # TESTING THE OBJECT-ORIENTED PIPELINE
    # ==========================================
    # Suppose we have a 3x3 matrix where Row 2 is a duplicate (2x) of Row 1
    raw_data = [
        [1, 2, 3],
        [2, 4, 6],  # FAKE DIMENSION
        [3, 1, 5]
    ]
    
    # Create the original Matrix object
    m1 = Matrix(3, 3, data=raw_data)
    
    # Pass the object into our engine
    computed_rank, clean_matrix_obj = get_row_echelon_matrix(m1)
    
    print("\n=== ENGINE RESULTS ===\n")
    print(f"Matrix Rank: {computed_rank}")
    
    print("\n=== FILTERED MATRIX OBJECT DATA ===")
    # Proving that it is a fully functional Matrix object
    print(f"New Object Rows: {clean_matrix_obj.row}") # Should be 2
    print(f"New Object Cols: {clean_matrix_obj.col}") # Should be 3
    print("Clean Data Grid:")
    for row in clean_matrix_obj.matrix:
            print(row)

    # ==========================================
    # BENCHMARKING & TESTING THE PIPELINE
    # ==========================================
    # Dataset with intentional redundancy
    # Row 1 and Row 3 are independent vectors.
    # Row 2 is exactly 2x of Row 1 (Linearly Dependent).
    dataset = [
        [1, 2, 3],
        [2, 4, 6],  
        [3, 1, 5]   
    ]
    
    m1 = Matrix(3, 3, data=dataset)
    
    # Execute the ML extraction engine
    computed_rank, final_ml_matrix = get_independent_features_matrix(m1)
    
    print("\n=== AI DATA PIPELINE RESULTS ===\n")
    print(f"Mathematical Rank: {computed_rank}")
    print("\nEXTRACTED INDEPENDENT FEATURES ->")
    print(f"Data Dimensions: {final_ml_matrix.row}x{final_ml_matrix.col}")
    
    # Notice how [3, 1, 5] is preserved exactly, not reduced to [0, -5, -4]
    print("Clean Data Grid:")
    for row in final_ml_matrix.matrix:
        print(row)

    # ==========================================
    # TESTING THE SVD ARCHITECTURE
    # ==========================================    
    dataset = [
        [3.0, 2.0],
        [2.0, 3.0],
        [6.0, 4.0],  
        [4.0, 6.0]   
    ]
    
    m1 = Matrix(4, 2, data=dataset)
    
    # Execute the Dimensionality Reduction Pipeline
    U, Sigma, V = get_dominant_svd(m1)
    
    print("\n=== SVD ENGINE RESULTS ===\n")
    print(f"Dominant Singular Value (Sigma): {Sigma:.4f}\n")
    
    print("Right Singular Vector (V) - Feature Alignment:")
    # V is a custom Matrix object (likely a column vector), so we extract its grid
    # and flatten it into a simple list for clean printing.
    print([val for val in V.matrix], "\n")
    
    print("Left Singular Vector (U) - User Importance:")
    # V is a custom Matrix object (likely a column vector), so we extract its grid
    # and flatten it into a simple list for clean printing.
    print([val for val in U.matrix])

    # =====================================================================
    # TESTING: PSEUDO-INVERSE ENGINES (NORMAL VS SVD)
    # =====================================================================
    print("\nPSEUDO-INVERSE TESTING...\n")

    # A Rectangular Matrix (3 Rows, 2 Columns)
    test_data = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    
    m_rect = Matrix(3, 2, data=test_data)
    
    print("--- ORIGINAL RECTANGULAR MATRIX (3x2) ---")
    print(m_rect)
    print("\n" + "="*50 + "\n")

    # ---------------------------------------------------------
    # TEST 1: NORMAL EQUATION PSEUDO-INVERSE (A^T * A)^-1 * A^T
    # ---------------------------------------------------------
    print("--- 1. NORMAL EQUATION PSEUDO-INVERSE ---")
    try:
        pseudo_inv_normal = get_pseudo_inverse(m_rect)
        print("✅ Calculated Pseudo-Inverse (Normal Method):")
        
        print(pseudo_inv_normal)
            
    except Exception as e:
        print(f"❌ Error in Normal Pseudo-Inverse: {e}")

    print("\n" + "="*50 + "\n")

    # ---------------------------------------------------------
    # TEST 2: TRUE SVD PSEUDO-INVERSE (V * Sigma^+ * U^T)
    # ---------------------------------------------------------
    print("--- 2. TRUE SVD PSEUDO-INVERSE ---")
    try:
        pseudo_inv_svd = get_pseudo_inverse_svd(m_rect)
        print("✅ Calculated Pseudo-Inverse (SVD Method):")

        print(pseudo_inv_svd)
            
    except Exception as e:
        print(f"❌ Error in SVD Pseudo-Inverse: {e}")

    print("\n🎯 TESTING COMPLETE.")