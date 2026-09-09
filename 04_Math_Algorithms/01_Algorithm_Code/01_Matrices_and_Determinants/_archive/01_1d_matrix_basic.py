import random

class Matrix:
    """
    An early version (v1) of a Matrix engine using a flattened 1D list structure.
    
    This class represents a 2D matrix by storing all elements in a single 1D list 
    and uses mathematical slicing to display and manipulate them.
    
    Attributes:
        col (int): The number of columns in the matrix.
        row (int): The number of rows in the matrix.
        num (list): A 1D list containing all the matrix elements.
    """
    def __init__(self, m, n, data=None):
        """
        Initializes the legacy Matrix object.

        Args:
            m (int): Number of rows.
            n (int): Number of columns.
            data (list, optional): A predefined 1D list to initialize the matrix. Defaults to None.
        """
        self.col = n
        self.row = m

        if data is not None:
            self.num = data

        else:
            # Generates a flat list of random integers between -5 and 5
            self.num = [random.randint(-5, 5) for i in range(m*n)]

    def __str__(self):
        """
        Formats the 1D list into a 2D grid view for terminal output.

        Logic:
            Slices the 1D list into chunks of size 'col' to represent individual rows.

        Returns:
            str: The formatted string representation of the matrix.
        """
        matrix_view = '\n'
        for i in range(0, self.col*self.row, self.col):
            row = [f"{element:^3}" for element in self.num[i: i+self.col]]
            matrix_view += '[' + ' '.join(row) + "]\n"
        return matrix_view
    
    def __add__(self, other):
        """
        Adds two matrices using flat 1D list iteration.

        Args:
            other (Matrix): The other Matrix object to add.

        Returns:
            Matrix or str: A new Matrix object containing the sum, or an error message if dimensions differ.
        """
        if self.col == other.col and self.row == other.row:
            sum_matrix = []
            for i in range(len(self.num)):
                sum_matrix.append(self.num[i] + other.num[i])

            return Matrix(self.row, self.col, data=sum_matrix)
        
        else:
            return 'To add the two matrices above, their order must be the same.'
        
    def __sub__(self, other):
        """
        Subtracts another matrix from the current matrix using flat 1D list iteration.

        Args:
            other (Matrix): The other Matrix object to subtract.

        Returns:
            Matrix or str: A new Matrix object containing the difference, or an error message if dimensions differ.
        """
        if self.col == other.col and self.row == other.row:
            sub_matrix = []
            for i in range(len(self.num)):
                sub_matrix.append(self.num[i] - other.num[i])

            return Matrix(self.row, self.col, data=sub_matrix)
            
        else:
            return 'To substract the two matrices above, their order must be the same.'

if __name__ == "__main__":
    m1 = Matrix(2, 3)
    m2 = Matrix(2, 3)
    print(m1)
    print(m2)

    print("Addition of Metrices -")
    print(m1 + m2)

    print("Substraction of Metrices -")
    print(m1 - m2)