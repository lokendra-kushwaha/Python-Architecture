from matrix_operation import Matrix
    
def get_determinant(matrix):
    """
    Calculates the determinant of 1x1, 2x2, and 3x3 matrices using direct formulas.

    Logic: 
        - Validates if the matrix is square.
        - 1x1 Edge Case: Returns the single element directly.
        - For 2x2: Applies formula (a11*a22) - (a12*a21).
        - For 3x3: Expands along the first row calculating the minor for each element.

    Returns:
        int, float, or str: The determinant value, or an error string if the matrix is not square.
    """
    if matrix.col != matrix.row:
        raise ValueError("Determinant can not be find.")
    
    if matrix.col == 1 and matrix.row == 1:
        return matrix[0][0]
        
    if matrix.col == 2 and matrix.row == 2:
        a11, a12 = matrix[0][0], matrix[0][1]
        a21, a22 = matrix[1][0], matrix[1][1]

        return (a11 * a22) - (a12 * a21)
    
    if matrix.col == 3 and matrix.row == 3:
        a11, a12, a13 = matrix[0][0], matrix[0][1], matrix[0][2]
        a21, a22, a23 = matrix[1][0], matrix[1][1], matrix[1][2]
        a31, a32, a33 = matrix[2][0], matrix[2][1], matrix[2][2]

        i = (a22 * a33) - (a23 * a32)
        j = (a21 * a33) - (a23 * a31)
        k = (a21 * a32) - (a22 * a31)

        return (a11*i - a12*j + a13*k)

    else:
        raise ValueError(f"Cannot calculate determinant for {matrix.row}x{matrix.col} matrix. Maximum supported order is 3x3.")


def get_inverse(matrix):
    """
    Calculates the Inverse of 1x1, 2x2, and 3x3 matrices using the Adjoint method.

    Step-by-Step Logic:
        1. Square Check: Ensures the matrix is an n x n square.
        2. 1x1 Matrix: Calculates the reciprocal of the single element.
        3. Hardcoded Unpacking: Maps matrix elements to variables for fast O(1) processing.
        4. Determinant Check: If the determinant is 0, it returns a singularity error.
        5. Cofactors & Adjoint: Calculates cofactors and transposes them to form the Adjoint matrix.
        6. Final Inverse: Multiplies every element in adjA by (1 / det) and rounds to 2 decimals.

    Returns:
        Matrix or str: A new Matrix object containing the inversed matrix data, 
                        or an error string if the matrix is singular or non-square.
    """
    if matrix.col != matrix.row:
        raise ValueError("Non-singular of matrix doesn't exist. (Not a square matrix)")
    
    if matrix.row == 1 and matrix.col == 1:
        if matrix[0][0] == 0:
            raise ValueError("This is a singular Matrix. (Determinant = 0)")

        inverse_matrix = [[(1/matrix[0][0])]]
        return Matrix(matrix.row, matrix.col, data=inverse_matrix)
        
    if matrix.col == 2 and matrix.row == 2:
        a11, a12 = matrix[0][0], matrix[0][1]
        a21, a22 = matrix[1][0], matrix[1][1]

        det=  (a11 * a22) - (a12 * a21)
        if det == 0:
            raise ValueError("This is a singular Matrix. (Determinant = 0)")

        A11, A12 = a22, -a21
        A21, A22 = -a12, a11
        adjA = [[A11, A21], [A12, A22]]

        inverse_matrix = [[(element*1/det) for element in row] for row in adjA]
        # for rows in adjA:
        #     new_row = []
        #     for element in rows:
        #         new_row.append(element*1/det)

        #     inverse_Matrix.append(new_row)
        return Matrix(matrix.row, matrix.col, data=inverse_matrix)
        
    if matrix.col == 3 and matrix.row == 3:
        a11, a12, a13 = matrix[0][0], matrix[0][1], matrix[0][2]
        a21, a22, a23 = matrix[1][0], matrix[1][1], matrix[1][2]
        a31, a32, a33 = matrix[2][0], matrix[2][1], matrix[2][2]

        A11, A12, A13 = (a22*a33 - a23*a32), -(a21*a33 - a23*a31), (a21*a32 - a22*a31)
        A21, A22, A23 = -(a12*a33 - a13*a32), (a11*a33 - a13*a31), -(a11*a32 - a12*a31)
        A31, A32, A33 = (a12*a23 - a13*a22), -(a11*a23 - a13*a21), (a11*a22 - a12*a21)

        det = a11*A11 + a12*A12 + a13*A13
        if det == 0:
            raise ValueError("This is a singular Matrix. (Determinant = 0)")
        
        adjA = [[A11, A21, A31], [A12, A22, A32], [A13, A23, A33]]

        inverse_Matrix = [[round(element*1/det, 2) for element in row] for row in adjA]
        
        return Matrix(matrix.row, matrix.col, data=inverse_Matrix)

    else:
        raise TypeError(f"Cannot calculate inverse matrix for {matrix.row}x{matrix.col} matrix. Maximum supported order is 3x3.")


if __name__ == "__main__":
    print("=" * 50)
    print("          DETERMINANT & INVERSE TESTING           ")
    print("=" * 50)

    m1 = Matrix(2, 2)
    m2 = Matrix(3, 3)
    
    print("\n[ INITIAL MATRICES ]")
    print("-" * 50)
    print("Matrix 1 (2x2) (m1):")
    print(m1)
    
    print("\nMatrix 2 (3x3) (m2):")
    print(m2)

    print("\n" + "=" * 50)
    print("                 DETERMINANTS                     ")
    print("=" * 50)
    
    print(f"Determinant of m1 (2x2) : {get_determinant(m1)}")
    print(f"Determinant of m2 (3x3) : {get_determinant(m2)}")

    print("\n" + "=" * 50)
    print("               INVERSE MATRICES                   ")
    print("=" * 50)
    
    print("Inverse of 1st matrix (m1) -->")
    print(get_inverse(m1))

    print("\nInverse of 2nd matrix (m2) -->")
    print(get_inverse(m2))

    print("\n" + "=" * 50)
    print("                    TESTING DONE                  ")
    print("=" * 50)

    