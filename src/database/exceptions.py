"""
Custom database exceptions for professional error handling.

All database-related exceptions inherit from DatabaseError for easy catching.
"""


class DatabaseError(Exception):
    """Base exception for all database-related errors."""
    
    def __init__(self, message: str, original_error: Exception = None):
        """
        Initialize database error.
        
        Args:
            message: Human-readable error message
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
    
    def __str__(self) -> str:
        """Return error message with original error if available."""
        if self.original_error:
            return f"{self.message} (Original: {type(self.original_error).__name__}: {str(self.original_error)})"
        return self.message


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when a requested record is not found."""
    
    def __init__(self, model_name: str, identifier: str, original_error: Exception = None):
        """
        Initialize record not found error.
        
        Args:
            model_name: Name of the model/table
            identifier: Identifier used to search for the record
            original_error: Original exception
        """
        message = f"{model_name} with identifier '{identifier}' not found"
        super().__init__(message, original_error)
        self.model_name = model_name
        self.identifier = identifier


class DuplicateRecordError(DatabaseError):
    """Raised when attempting to create a duplicate record."""
    
    def __init__(self, model_name: str, field: str, value: str, original_error: Exception = None):
        """
        Initialize duplicate record error.
        
        Args:
            model_name: Name of the model/table
            field: Field that has duplicate value
            value: Duplicate value
            original_error: Original exception
        """
        message = f"{model_name} with {field}='{value}' already exists"
        super().__init__(message, original_error)
        self.model_name = model_name
        self.field = field
        self.value = value


class DatabaseTransactionError(DatabaseError):
    """Raised when a database transaction fails."""
    pass
