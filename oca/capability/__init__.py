"""Typed provider capability primitives used natively by OCA."""
from .catalog import Capability, CapabilityCatalog, CapabilityError
from .provider import ProviderSemantics, get_provider_semantics
from .types import (
    FieldTypeError, SemanticField, SEMANTIC_FIELDS, canonical_field,
    compatible_operands, resolve_field,
)

__all__ = [
    "Capability", "CapabilityCatalog", "CapabilityError",
    "ProviderSemantics", "get_provider_semantics",
    "FieldTypeError", "SemanticField", "SEMANTIC_FIELDS",
    "canonical_field", "compatible_operands", "resolve_field",
]
