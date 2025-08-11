"""
Custom exceptions for the MLOps project.
"""

class SandyieException(Exception):
    """
    Custom exception class for MLOps pipeline errors.
    """
    def __init__(self, message="An error occurred in the MLOps pipeline."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"SandyieException: {self.message}"

# Example usage:
# try:
#     raise SandyieException("Data file not found.")
# except SandyieException as e:
#     print(e)