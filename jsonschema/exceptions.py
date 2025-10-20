class ValidationError(ValueError):
    """Exception raised when JSON Schema validation fails."""

    def __init__(self, message, path=None):
        super().__init__(message)
        self.path = path or []

    def __str__(self):
        if not self.path:
            return super().__str__()
        joined = ".".join(str(p) for p in self.path)
        return f"{super().__str__()} (at {joined})"
