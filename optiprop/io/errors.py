"""Exceptions raised at the OptiProp field-I/O boundary."""


class FieldIOError(Exception):
    """Base class for field inspection/import/export failures."""


class UnsupportedFormatError(FieldIOError):
    """The file suffix or explicit format is unsupported."""


class SchemaValidationError(FieldIOError):
    """A claimed canonical dataset violates its schema."""


class AmbiguousMappingError(FieldIOError):
    """A noncanonical dataset requires an explicit import mapping."""


class ImportMappingError(FieldIOError):
    """An explicit import mapping is incomplete or incompatible with data."""


class UnsafeDatasetError(FieldIOError):
    """A dataset requires pickle/object loading or another unsafe operation."""


__all__ = [
    "AmbiguousMappingError",
    "FieldIOError",
    "ImportMappingError",
    "SchemaValidationError",
    "UnsafeDatasetError",
    "UnsupportedFormatError",
]
