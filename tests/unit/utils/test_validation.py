import pytest
from novelentitymatcher.exceptions import ValidationError
from novelentitymatcher.utils.validation import (
    validate_entity,
    validate_entities,
    validate_threshold,
    validate_model_name,
)


class TestValidation:
    """Tests for input validation utilities."""

    def test_validate_entity_valid(self):
        entity = {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]}
        assert validate_entity(entity) is True

    def test_validate_entity_missing_id(self):
        entity = {"name": "Germany"}
        with pytest.raises(ValueError, match="must have 'id'"):
            validate_entity(entity)

    def test_validate_entity_missing_id_raises_validation_error(self):
        with pytest.raises(ValidationError):
            validate_entity({"name": "Germany"})

    def test_validate_entity_missing_name(self):
        entity = {"id": "DE"}
        with pytest.raises(ValueError, match="must have 'name'"):
            validate_entity(entity)

    def test_validate_entities_valid(self):
        entities = [
            {"id": "DE", "name": "Germany"},
            {"id": "FR", "name": "France"},
        ]
        assert validate_entities(entities) is True

    def test_validate_entities_empty(self):
        with pytest.raises(ValueError, match="entities list cannot be empty"):
            validate_entities([])

    def test_validate_entities_duplicate_ids_raise_validation_error(self):
        with pytest.raises(ValidationError, match="duplicates"):
            validate_entities(
                [
                    {"id": "DE", "name": "Germany"},
                    {"id": "DE", "name": "Deutschland"},
                ]
            )

    def test_validate_threshold_valid(self):
        assert validate_threshold(0.5) == 0.5
        assert validate_threshold(0.0) == 0.0
        assert validate_threshold(1.0) == 1.0

    def test_validate_threshold_invalid(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_threshold(1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_threshold(-0.1)

    def test_validate_model_name_valid(self):
        assert validate_model_name("sentence-transformers/paraphrase-mpnet-base-v2")
        assert validate_model_name("BAAI/bge-m3")

    def test_validate_model_name_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_model_name("")
