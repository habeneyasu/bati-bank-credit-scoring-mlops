"""
Unit tests for PII redaction functionality.

Tests verify PII detection, redaction strategies, and integration with logging.

Run with: pytest tests/test_pii_redaction.py -v
"""

import pytest
from src.utils.pii_redaction import (
    PIIRedactor,
    RedactionStrategy,
    PIIType,
    redact_pii,
    get_redactor
)


class TestPIIRedactor:
    """Tests for PIIRedactor class."""
    
    def test_email_detection(self):
        """Test email detection in text."""
        redactor = PIIRedactor()
        text = "Contact us at support@example.com for help"
        detected = redactor.detect_pii(text)
        assert PIIType.EMAIL in detected
    
    def test_phone_detection(self):
        """Test phone number detection."""
        redactor = PIIRedactor()
        text = "Call us at 555-123-4567"
        detected = redactor.detect_pii(text)
        assert PIIType.PHONE in detected
    
    def test_ssn_detection(self):
        """Test SSN detection."""
        redactor = PIIRedactor()
        text = "SSN: 123-45-6789"
        detected = redactor.detect_pii(text)
        assert PIIType.SSN in detected
    
    def test_ip_address_detection(self):
        """Test IP address detection."""
        redactor = PIIRedactor()
        text = "IP: 192.168.1.1"
        detected = redactor.detect_pii(text)
        assert PIIType.IP_ADDRESS in detected
    
    def test_mask_strategy_email(self):
        """Test masking strategy for email."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        email = "john.doe@example.com"
        redacted = redactor.redact_value(email, PIIType.EMAIL)
        assert "@" in redacted
        assert "***" in redacted
        assert "john.doe" not in redacted
        assert "example.com" not in redacted
    
    def test_mask_strategy_phone(self):
        """Test masking strategy for phone."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        phone = "555-123-4567"
        redacted = redactor.redact_value(phone, PIIType.PHONE)
        assert "***" in redacted
        assert "4567" in redacted  # Last 4 digits shown
    
    def test_hash_strategy(self):
        """Test hashing strategy."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.HASH)
        email = "test@example.com"
        redacted = redactor.redact_value(email, PIIType.EMAIL)
        assert redacted.startswith("sha256:")
        assert len(redacted) > 20
    
    def test_remove_strategy(self):
        """Test removal strategy."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.REMOVE)
        email = "test@example.com"
        redacted = redactor.redact_value(email, PIIType.EMAIL)
        assert redacted is None
    
    def test_partial_mask_strategy(self):
        """Test partial masking strategy."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.PARTIAL_MASK)
        email = "john.doe@example.com"
        redacted = redactor.redact_value(email, PIIType.EMAIL)
        assert "@" in redacted
        assert "example.com" in redacted  # Domain preserved
        assert "j***" in redacted  # Local part masked
    
    def test_redact_dict_email_field(self):
        """Test redacting email from dictionary."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        data = {
            "user_id": 123,
            "email": "user@example.com",
            "name": "John Doe"
        }
        redacted = redactor.redact_dict(data)
        assert redacted["user_id"] == 123  # Not PII
        assert redacted["email"] != "user@example.com"  # Redacted
        assert "***" in redacted["email"]
        assert redacted["name"] == "John Doe"  # Not in default PII fields
    
    def test_redact_dict_customer_id(self):
        """Test redacting customer_id from dictionary."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        data = {
            "customer_id": "CUST-12345",
            "prediction": 0,
            "probability": 0.15
        }
        redacted = redactor.redact_dict(data)
        assert redacted["customer_id"] != "CUST-12345"  # Redacted
        assert redacted["prediction"] == 0  # Not PII
        assert redacted["probability"] == 0.15  # Not PII
    
    def test_redact_nested_dict(self):
        """Test redacting PII from nested dictionary."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        data = {
            "user": {
                "email": "user@example.com",
                "phone": "555-123-4567"
            },
            "metadata": {
                "customer_id": "CUST-123"
            }
        }
        redacted = redactor.redact_dict(data, recursive=True)
        assert "***" in redacted["user"]["email"]
        assert "***" in redacted["user"]["phone"]
        assert redacted["metadata"]["customer_id"] != "CUST-123"
    
    def test_redact_list(self):
        """Test redacting PII from list."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        data = [
            {"email": "user1@example.com", "id": 1},
            {"email": "user2@example.com", "id": 2}
        ]
        redacted = redactor.redact_list(data, recursive=True)
        assert len(redacted) == 2
        assert "***" in redacted[0]["email"]
        assert "***" in redacted[1]["email"]
        assert redacted[0]["id"] == 1
        assert redacted[1]["id"] == 2
    
    def test_custom_field_mapping(self):
        """Test custom field mapping."""
        custom_fields = {"user_email": PIIType.EMAIL}
        redactor = PIIRedactor(
            redaction_strategy=RedactionStrategy.MASK,
            custom_fields=custom_fields
        )
        data = {"user_email": "test@example.com"}
        redacted = redactor.redact_dict(data)
        assert redacted["user_email"] != "test@example.com"
        assert "***" in redacted["user_email"]
    
    def test_selective_redaction(self):
        """Test redacting only specific PII types."""
        redactor = PIIRedactor(
            redaction_strategy=RedactionStrategy.MASK,
            redact_types=[PIIType.EMAIL]  # Only redact emails
        )
        data = {
            "email": "user@example.com",
            "phone": "555-123-4567",
            "customer_id": "CUST-123"
        }
        redacted = redactor.redact_dict(data)
        assert "***" in redacted["email"]  # Redacted
        assert redacted["phone"] == "555-123-4567"  # Not redacted
        assert redacted["customer_id"] == "CUST-123"  # Not redacted
    
    def test_redact_json_string(self):
        """Test redacting PII from JSON string."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        json_str = '{"email": "user@example.com", "customer_id": "CUST-123"}'
        redacted = redactor.redact_json(json_str)
        assert "user@example.com" not in redacted
        assert "CUST-123" not in redacted
        assert "***" in redacted


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_redact_pii_function(self):
        """Test redact_pii convenience function."""
        data = {"email": "user@example.com", "id": 123}
        redacted = redact_pii(data)
        assert "user@example.com" not in str(redacted)
    
    def test_get_redactor(self):
        """Test get_redactor function."""
        redactor = get_redactor()
        assert isinstance(redactor, PIIRedactor)


class TestIntegration:
    """Integration tests for PII redaction."""
    
    def test_log_data_redaction(self):
        """Test redacting log data structure."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        log_data = {
            "customer_id": "CUST-12345",
            "email": "user@example.com",
            "prediction": 0,
            "probability": 0.15,
            "metadata": {
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0"
            }
        }
        redacted = redactor.redact_dict(log_data, recursive=True)
        assert redacted["prediction"] == 0  # Not PII
        assert redacted["probability"] == 0.15  # Not PII
        assert "CUST-12345" not in str(redacted["customer_id"])
        assert "user@example.com" not in str(redacted["email"])
        assert "192.168.1.1" not in str(redacted["metadata"]["ip_address"])
    
    def test_api_response_redaction(self):
        """Test redacting API response data."""
        redactor = PIIRedactor(redaction_strategy=RedactionStrategy.MASK)
        response_data = {
            "customer_id": "CUST-12345",
            "prediction": 0,
            "probability": 0.15,
            "user": {
                "email": "user@example.com",
                "full_name": "John Doe"
            }
        }
        redacted = redactor.redact_dict(response_data, recursive=True)
        assert redacted["prediction"] == 0
        assert redacted["probability"] == 0.15
        assert "CUST-12345" not in str(redacted["customer_id"])
        assert "user@example.com" not in str(redacted["user"]["email"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
