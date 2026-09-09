"""
=============================================================================
Linear Algebra Framework - Core Matrix Operations
=============================================================================

A pure Python, independent implementation of a 2D Matrix data structure.
This module serves as the foundational "Motherboard" for advanced mathematical 
computations, built entirely from scratch without relying on external libraries 
like NumPy or SciPy.

It handles memory layout, strict type validation, dynamic reshaping, and 
provides a comprehensive suite of matrix operations using both explicit 
functional methods and Pythonic magic methods (operator overloading).

Key Capabilities & Architecture:
--------------------------------
1. Dynamic Instantiation & Generators: 
   - Auto-reshapes 1D flat lists into C-style 2D grids.
   - Validates explicit 2D nested lists and supports randomized generation.
   - Built-in class methods for Identity and Zero matrices.

2. Operator Overloading (Magic Methods):
   - Element-wise Arithmetic: +, -, *, / 
   - Advanced Operations: -A (Negation), A ** n (Exponentiation), A == B (Equality)
   - NumPy-Style Indexing: A[row, col] direct access and assignment.

3. Algebraic & Scalar Operations:
   - Matrix Multiplication (Dot Product).
   - Scalar Multiplication and Division.
   - Explicit functional fallback methods (.add(), .subtract()).

4. Transformations & Mathematical Properties:
   - Fast matrix transposition (A^T).
   - Boolean validation for Symmetric and Skew-Symmetric matrices.

Design Philosophy:
------------------
- First Principles: Built from scratch to deeply understand the underlying 
  mathematics and memory manipulation of standard AI numerical libraries.
- Production-Ready: Implements strict edge-case handling, dimension 
  validation (ValueErrors), and highly optimized list comprehensions for 
  maximum execution speed in pure Python.
=============================================================================
"""

from random import randint

class Matrix:
    """
    =============================================================================
    Core Matrix Object 
    =============================================================================
    
    A foundational class representing a mathematical 2D Matrix. 
    This acts as the primary data structure for the entire custom linear algebra 
    engine, handling memory layout, data validation, and dynamic reshaping 
    without relying on external libraries like NumPy.

    Attributes:
        row (int): The number of rows in the matrix.
        col (int): The number of columns in the matrix.
        matrix (list): A nested 2D list containing the raw numerical data.
    """

    def __init__(self, row, col, data=None):
        """
        Initializes the Matrix object and constructs the internal 2D grid.
        
        It dynamically handles three types of instantiation:
        1. 2D List: Direct mapping if dimensions match.
        2. 1D Flat List: Automatically reshapes C-style flat arrays into a 2D grid.
        3. None: Auto-generates a matrix with random integers between -10 and 10.

        Args:
            row (int): The required number of rows.
            col (int): The required number of columns.
            data (list, optional): The raw data to populate the matrix. Can be a 
                                   1D flat list or a 2D nested list. Defaults to None.
                                   
        Raises:
            TypeError: If the provided data is not a list.
            ValueError: If the dimensions of the provided data do not match 
                        the specified 'row' and 'col' arguments.
        """
        self.row = row
        self.col = col
        
        # ---------------------------------------------------------
        # SCENARIO A: USER PROVIDED SPECIFIC DATA
        # ---------------------------------------------------------
        if data is not None:
            # 1. Strict Type Validation
            if not isinstance(data, list):
                raise TypeError(f"Matrix data must be a list, got {type(data).__name__} instead.")

            # 2. Handling 2D Nested Lists
            if len(data) > 0 and isinstance(data[0], list):
                # Validate that the number of rows matches, and EVERY row has the correct number of columns
                if len(data) != self.row or any(len(r) != self.col for r in data):
                    raise ValueError(f"Dimension mismatch! Expected a {self.row}x{self.col} grid.")
                
                self.matrix = data

            # 3. Handling 1D Flat Lists (The Reshape Engine)
            else:
                # Validate that the flat list has the exact total number of elements required
                if (len(data) % self.row != 0) or (self.row * self.col != len(data)):
                    raise ValueError(f"Cannot reshape array of size {len(data)} into shape ({self.row}, {self.col}).")

                matrix = []
                i = 0
                
                # Slicing the 1D array into discrete rows (Memory mapping logic)
                while i < self.row:
                    # Extracts a chunk of size 'col' for the current row
                    split = data[i * self.col : self.col + self.col * i]
                    matrix.append(split)
                    i += 1
                    
                self.matrix = matrix

        # ---------------------------------------------------------
        # SCENARIO B: NO DATA PROVIDED (Random Generation Fallback)
        # ---------------------------------------------------------
        else:
            # Uses list comprehension to rapidly generate a randomized 2D grid
            self.matrix = [[randint(-10, 10) for _ in range(self.col)] for _ in range(self.row)]

    # =====================================================================
    # MATRIX GENERATORS (CLASS METHODS)
    # =====================================================================
    @classmethod
    def identity(cls, n):
        """
        Generates an N x N Identity Matrix (1s on the diagonal, 0s elsewhere).
        Usage: m = Matrix.identity(n)
        """
        matrix_data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(n, n, data=matrix_data)

    @classmethod
    def zeros(cls, row, col):
        """
        Generates a matrix filled entirely with zeros.
        Usage: m = Matrix.zeros(row, col)
        """
        matrix_data = [[0 for _ in range(col)] for _ in range(row)]
        return cls(row, col, data=matrix_data)

    def __str__(self):
        """
        Returns a string representation of the matrix for easy printing and readability.
        
        Returns:
            str: The matrix formatted row by row with a separator line at the end.
        """
        display_matrix = "\n".join(["[" + ", ".join([f"{val:.2f}" for val in row]) + "]" for row in self.matrix])

        return display_matrix

    # =====================================================================
    # ADVANCED INDEXING (NUMPY STYLE)
    # =====================================================================
    def __getitem__(self, index):
        """Allows NumPy-style indexing: matrix[row, col] instead of matrix.matrix[row][col]"""
        if isinstance(index, tuple) and len(index) == 2:
            r, c = index
            return self.matrix[r][c]
        return self.matrix[index] # If only row index is provided

    def __setitem__(self, index, value):
        """Allows assigning values directly: matrix[row, col] = 5"""
        if isinstance(index, tuple) and len(index) == 2:
            r, c = index
            self.matrix[r][c] = value
        else:
            self.matrix[index] = value

    # =====================================================================
    # ELEMENT-WISE ARITHMETIC OPERATIONS (MAGIC METHODS)
    # =====================================================================

    def __add__(self, other):
        """
        Performs element-wise matrix addition using the '+' operator.
        Both matrices must have the exact same dimensions.

        Args:
            other (Matrix): The right-hand operand matrix to be added.

        Returns:
            Matrix: A new Matrix object containing the element-wise sum.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for addition: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        # Pythonic optimization: Using nested list comprehension for faster execution
        sum_matrices = [
            [self.matrix[i][j] + other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]
        
        return Matrix(self.row, self.col, data=sum_matrices)

    def __sub__(self, other):
        """
        Performs element-wise matrix subtraction using the '-' operator.
        Both matrices must have the exact same dimensions.

        Args:
            other (Matrix): The right-hand operand matrix to be subtracted.

        Returns:
            Matrix: A new Matrix object containing the element-wise difference.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for subtraction: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        sub_matrices = [
            [self.matrix[i][j] - other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=sub_matrices)

    def __mul__(self, other):
        """
        Performs ELEMENT-WISE matrix multiplication (Hadamard Product) using the '*' operator.
        
        IMPORTANT ARCHITECTURE NOTE: 
        This is NOT the algebraic Matrix Dot Product. This simply multiplies 
        corresponding elements at [i][j]. For Dot Product, use the .multiply() method.

        Args:
            other (Matrix): The right-hand operand matrix.

        Returns:
            Matrix: A new Matrix object containing the element-wise product.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for element-wise multiplication: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        mul_matrices = [
            [self.matrix[i][j] * other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=mul_matrices)

    def __truediv__(self, other):
        """
        Performs element-wise matrix division using the '/' operator.
        Both matrices must have the exact same dimensions.

        Args:
            other (Matrix): The denominator matrix.

        Returns:
            Matrix: A new Matrix object containing the element-wise quotient.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
            ZeroDivisionError: Handled natively by Python if any element in 'other' is 0.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for division: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        div_matrices = [
            [self.matrix[i][j] / other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=div_matrices)

    # =====================================================================
    # ADVANCED MAGIC METHODS (NEGATION, POWER, EQUALITY)
    # =====================================================================

    def __neg__(self):
        """
        Returns the negation of the matrix (-A).
        Every element in the matrix is multiplied by -1.
        
        Usage: m2 = -m1

        Returns:
            Matrix: A new Matrix object with inverted signs.
        """
        # Reusing the scalar engine.
        return self.scalar_multiply(-1)

    def __pow__(self, exponent):
        """
        Performs ELEMENT-WISE exponentiation.
        Raises each element in the matrix to the power of the given exponent.
        
        Usage: m2 = m1 ** 2

        Args:
            exponent (int or float): The power to raise the elements to.

        Returns:
            Matrix: A new Matrix object with exponentiated values.
            
        Raises:
            TypeError: If the exponent provided is not a numerical value (int or float).
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be an integer or float, got {type(exponent).__name__} instead.")

        pow_matrix = [
            [self.matrix[i][j] ** exponent for j in range(self.col)]
            for i in range(self.row)
        ]
        
        return Matrix(self.row, self.col, data=pow_matrix)

    def __eq__(self, other):
        """
        Checks if two matrices are strictly equal in dimensions and data.
        
        Usage: if m1 == m2:

        Args:
            other (Matrix): The right-hand object to compare with.

        Returns:
            bool: True if dimensions and all elements match exactly, False otherwise.
        """
        if not isinstance(other, Matrix):
            return False

        return (self.row == other.row) and (self.col == other.col) and (self.matrix == other.matrix)

    # =====================================================================
    # EXPLICIT FUNCTIONAL ARITHMETIC METHODS
    # =====================================================================

    def add(self, other):
        """
        Performs explicit element-wise matrix addition. 
        Both matrices must have the exact same dimensions (order).

        This is the functional equivalent of the '+' operator (__add__).

        Args:
            other (Matrix): The matrix to be added to the current matrix.

        Returns:
            Matrix: A new Matrix object containing the element-wise sum.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for addition: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        sum_matrices = [
            [self.matrix[i][j] + other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=sum_matrices)

    def subtract(self, other):
        """
        Performs explicit element-wise matrix subtraction.
        Both matrices must have the exact same dimensions (order).

        This is the functional equivalent of the '-' operator (__sub__).

        Args:
            other (Matrix): The matrix to be subtracted from the current matrix.

        Returns:
            Matrix: A new Matrix object containing the element-wise difference.
            
        Raises:
            ValueError: If the dimensions (order) of the two matrices do not match.
        """
        if self.col != other.col or self.row != other.row:
            raise ValueError(f"Dimension mismatch for subtraction: ({self.row}x{self.col}) and ({other.row}x{other.col}).")

        sub_matrices = [
            [self.matrix[i][j] - other.matrix[i][j] for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=sub_matrices)

    # =====================================================================
    # MATRIX DOT PRODUCT & SCALAR OPERATIONS
    # =====================================================================

    def multiply(self, other):
        """
        Performs algebraic matrix multiplication (Dot Product).
        
        The number of columns in the current matrix (self.col) MUST equal 
        the number of rows in the second matrix (other.row).

        Args:
            other (Matrix): The right-hand operand matrix to multiply with.

        Returns:
            Matrix: A new Matrix object containing the dot product result.
            
        Raises:
            ValueError: If the dimensions are invalid for matrix multiplication.
        """
        if self.col != other.row:
            raise ValueError(f"Multiplication mismatch: Cannot multiply a ({self.row}x{self.col}) matrix with a ({other.row}x{other.col}) matrix.")
        
        # ---------------------------------------------------------
        # O(N^3) DOT PRODUCT ALGORITHM
        # ---------------------------------------------------------
        result_matrix = [
            [
                sum(self.matrix[i][k] * other.matrix[k][j] for k in range(self.col))
                for j in range(other.col)
            ]
            for i in range(self.row)
        ]

        return Matrix(self.row, other.col, data=result_matrix)

    def scalar_multiply(self, scalar):
        """
        Multiplies every element in the matrix by a given scalar value.

        Args:
            scalar (int or float): The numerical value to multiply the matrix by.

        Returns:
            Matrix: A new Matrix object with the scaled values.
        """
        scaled_matrix = [
            [self.matrix[i][j] * scalar for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=scaled_matrix)

    def scalar_divide(self, scalar):
        """
        Divides every element in the matrix by a given scalar value.

        Args:
            scalar (int or float): The numerical value (denominator).

        Returns:
            Matrix: A new Matrix object with the divided values.
            
        Raises:
            ZeroDivisionError: If the scalar provided is 0.
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a matrix by zero (scalar = 0).")

        divided_matrix = [
            [self.matrix[i][j] / scalar for j in range(self.col)]
            for i in range(self.row)
        ]

        return Matrix(self.row, self.col, data=divided_matrix)

    # =====================================================================
    # MATRIX TRANSFORMATIONS & PROPERTIES
    # =====================================================================

    def transpose(self):
        """
        Calculates the transpose of the matrix by swapping its rows and columns.
        (Converts an N x M matrix into an M x N matrix).

        Returns:
            Matrix: A new Matrix object that is the transpose of the original matrix.
        """
        transposed_matrix = [
            [self.matrix[i][j] for i in range(self.row)]
            for j in range(self.col)
        ]

        # Note we swap self.col and self.row in the new object initialization
        return Matrix(self.col, self.row, data=transposed_matrix)
    
    def is_symmetric(self):
        """
        Checks if the matrix is Symmetric. 
        A matrix is symmetric if it is a square matrix and strictly equals 
        its own transpose (A^T = A).

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        # A rectangular matrix can never be symmetric
        if self.col != self.row:
            return False

        return self.transpose() == self

    def is_skew_symmetric(self):
        """
        Checks if the matrix is Skew-Symmetric.
        A matrix is skew-symmetric if it is a square matrix and its transpose 
        equals its negative (A^T = -A).

        Returns:
            bool: True if the matrix is skew-symmetric, False otherwise.
        """
        if self.col != self.row:
            return False

        # Uses own scalar_multiply engine to dynamically calculate -A
        return self.transpose() == -(self)


if __name__ == "__main__":
    print("=" * 50)
    print("             MATRIX OPERATIONS TESTING            ")
    print("=" * 50)

    m1 = Matrix(2, 2)
    m2 = Matrix(2, 2)
    
    print("\n[ INITIAL MATRICES ]")
    print("Matrix 1 (m1):")
    print(m1)
    print("\nMatrix 2 (m2):")
    print(m2)

    print("\n" + "-" * 50)
    print("[ CLASS METHODS ]")
    print("-" * 50)

    print("\nIdentity Matrix (3x3) -->")
    m5 = Matrix.identity(3)
    print(m5)

    print("\nZeros Matrix (3x2) -->")
    m6 = Matrix.zeros(3, 2)
    print(m6)

    print("\n" + "-" * 50)
    print("[ MAGIC METHODS (+, -, *) ]")
    print("-" * 50)
    
    print("\nm1 - m2 -->")
    print(m1 - m2)
    
    print("\nm1 + m2 -->")
    print(m1 + m2)
    
    print("\nm1 * m2 -->")
    print(m1 * m2)

    print("\n" + "-" * 50)
    print("[ BASIC METHODS ]")
    print("-" * 50)

    print("\nAddition of matrices (m1.add(m2)) -->")
    print(m1.add(m2))

    print("\nSubtraction of matrices (m1.subtract(m2)) -->")
    print(m1.subtract(m2))

    print("\nMultiplication of matrices (m1.multiply(m2)) -->")
    print(m1.multiply(m2))

    print("\nScalar multiplication (m1.scalar_multiply(4)) -->")
    print(m1.scalar_multiply(4))

    print("\nScalar division (m1.scalar_divide(4)) -->")
    print(m1.scalar_divide(4))
    
    print("\nTranspose of matrix (m1.transpose()) -->")
    print(m1.transpose())

    print(f"\nIs m1 Symmetric? --> {m1.is_symmetric()}")

    m3 = Matrix(2, 2)
    print(f"Is m3 Skew-Symmetric? --> {m3.is_skew_symmetric()}")
    
    print("\n" + "=" * 50)
    print("                    TESTING DONE                  ")
    print("=" * 50)