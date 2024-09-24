__all__ = [
    'ParseError',
    'UnsupportedConstruct',
    'UnsupportedType',
    'AnnotationError',
    'UnknownPrimitiveOps',
    'ShapeError',
]


class ParseError(Exception):
    pass


class UnsupportedConstruct(ParseError):
    """Exception for unsupported Python construct."""

    def __init__(self, msg=None):
        self.msg = f"Unspport Python construct {msg}."


class UnsupportedType(ParseError):
    pass


class AnnotationError(ParseError):
    pass


class UnknownPrimitiveOps(ParseError):
    pass


class ShapeError(ParseError):
    pass
