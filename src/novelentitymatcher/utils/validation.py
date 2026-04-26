from typing import Any

from ..exceptions import ValidationError

__all__ = [
    "validate_entities",
    "validate_entity",
    "validate_model_name",
    "validate_threshold",
]


def validate_entity(entity: dict[str, Any]) -> bool:
    """Validate a single entity dictionary."""
    if "id" not in entity:
        raise ValidationError(
            "Entity must have 'id' field",
            entity=entity,
            field="id",
            suggestion="Add 'id' field: {'id': 'unique_id', 'name': 'Entity Name'}",
        )
    if "name" not in entity:
        raise ValidationError(
            "Entity must have 'name' field",
            entity=entity,
            field="name",
            suggestion=f"Add 'name' field: {{'id': '{entity.get('id', 'unknown')}', 'name': 'Entity Name'}}",
        )
    return True


def validate_entities(entities: list[dict[str, Any]]) -> bool:
    """Validate a list of entities."""
    if not entities:
        raise ValidationError(
            "entities list cannot be empty",
            suggestion="Provide at least one entity, e.g., [{'id': '1', 'name': 'Entity'}]",
        )

    for entity in entities:
        validate_entity(entity)

    ids = [e["id"] for e in entities]
    if len(ids) != len(set(ids)):
        # Find duplicate IDs
        seen = set()
        duplicates = []
        for eid in ids:
            if eid in seen:
                duplicates.append(eid)
            seen.add(eid)
        raise ValidationError(
            f"Entity IDs must be unique. Found duplicates: {duplicates}",
            suggestion="Ensure each entity has a unique ID. Check for typos or repeated entries.",
        )

    return True


def validate_threshold(threshold: float) -> float:
    """Validate similarity threshold value."""
    if not isinstance(threshold, (int, float)):
        raise ValidationError(
            f"threshold must be a number, got {type(threshold).__name__}",
            field="threshold",
            suggestion="Use a float between 0.0 and 1.0, e.g., threshold=0.7",
        )
    if threshold < 0 or threshold > 1:
        raise ValidationError(
            f"threshold must be between 0 and 1, got {threshold}",
            field="threshold",
            suggestion="Use a value between 0.0 (very permissive) and 1.0 (very strict)",
        )
    return float(threshold)


def validate_model_name(model_name: str) -> bool:
    """Validate model name is not empty."""
    if not model_name or not isinstance(model_name, str):
        raise ValidationError(
            "model_name cannot be empty",
            field="model_name",
            suggestion="Provide a valid model name, e.g., 'sentence-transformers/all-MiniLM-L6-v2'",
        )
    return True
