"""
PII (Personally Identifiable Information) Redaction Module

Provides PII detection and redaction capabilities for GDPR/CCPA compliance.
Supports multiple redaction strategies: masking, hashing, and removal.

This module is critical for:
- GDPR compliance (EU General Data Protection Regulation)
- CCPA compliance (California Consumer Privacy Act)
- Privacy protection in logs and audit trails
- Data minimization principles
"""

import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RedactionStrategy(Enum):
    """Redaction strategies for PII."""
    MASK = "mask"  # Replace with asterisks (e.g., "***@***.com")
    HASH = "hash"  # Replace with hash (e.g., "sha256:abc123...")
    REMOVE = "remove"  # Remove field entirely
    PARTIAL_MASK = "partial_mask"  # Show partial info (e.g., "j***@example.com")


class PIIType(Enum):
    """Types of PII that can be detected and redacted."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    CUSTOMER_ID = "customer_id"
    USERNAME = "username"
    FULL_NAME = "full_name"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    DATE_OF_BIRTH = "date_of_birth"


class PIIRedactor:
    """
    PII detection and redaction utility.
    
    Detects and redacts PII from dictionaries, strings, and nested structures.
    """
    
    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        PIIType.PHONE: re.compile(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ),
        PIIType.SSN: re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ),
        PIIType.IP_ADDRESS: re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ),
        PIIType.ZIP_CODE: re.compile(
            r'\b\d{5}(?:-\d{4})?\b'
        ),
        PIIType.DATE_OF_BIRTH: re.compile(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        ),
    }
    
    # Field names that likely contain PII
    PII_FIELD_NAMES = {
        PIIType.EMAIL: ['email', 'e_mail', 'email_address', 'email_addr'],
        PIIType.PHONE: ['phone', 'phone_number', 'mobile', 'telephone', 'tel'],
        PIIType.SSN: ['ssn', 'social_security', 'social_security_number'],
        PIIType.CREDIT_CARD: ['credit_card', 'card_number', 'cc_number', 'card'],
        PIIType.IP_ADDRESS: ['ip', 'ip_address', 'ip_addr', 'client_ip', 'remote_addr'],
        PIIType.CUSTOMER_ID: ['customer_id', 'customer', 'cust_id', 'user_id', 'client_id'],
        PIIType.USERNAME: ['username', 'user_name', 'login', 'account_name'],
        PIIType.FULL_NAME: ['full_name', 'name', 'first_name', 'last_name', 'fname', 'lname'],
        PIIType.ADDRESS: ['address', 'street_address', 'addr', 'location'],
        PIIType.ZIP_CODE: ['zip', 'zip_code', 'postal_code', 'postcode'],
        PIIType.DATE_OF_BIRTH: ['dob', 'date_of_birth', 'birth_date', 'birthdate'],
    }
    
    def __init__(
        self,
        redaction_strategy: RedactionStrategy = RedactionStrategy.MASK,
        redact_types: Optional[List[PIIType]] = None,
        custom_fields: Optional[Dict[str, PIIType]] = None,
        enable_logging: bool = True
    ):
        """
        Initialize PII redactor.
        
        Args:
            redaction_strategy: Strategy for redacting PII
            redact_types: List of PII types to redact (None = all)
            custom_fields: Custom field mappings (field_name -> PIIType)
            enable_logging: Whether to log redaction actions
        """
        self.redaction_strategy = redaction_strategy
        self.redact_types = redact_types or list(PIIType)
        self.custom_fields = custom_fields or {}
        self.enable_logging = enable_logging
        
        # Build field name lookup
        self.field_to_type = {}
        for pii_type, field_names in self.PII_FIELD_NAMES.items():
            for field_name in field_names:
                self.field_to_type[field_name.lower()] = pii_type
        # Add custom fields
        for field_name, pii_type in self.custom_fields.items():
            self.field_to_type[field_name.lower()] = pii_type
    
    def detect_pii(self, text: str) -> List[PIIType]:
        """
        Detect PII types in a text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected PII types
        """
        if not isinstance(text, str):
            return []
        
        detected = []
        for pii_type, pattern in self.PATTERNS.items():
            if pii_type in self.redact_types and pattern.search(text):
                detected.append(pii_type)
        
        return detected
    
    def redact_value(
        self,
        value: Any,
        pii_type: Optional[PIIType] = None,
        field_name: Optional[str] = None
    ) -> Any:
        """
        Redact a single value based on strategy.
        
        Args:
            value: Value to redact
            pii_type: Type of PII (if known)
            field_name: Field name (for type inference)
            
        Returns:
            Redacted value
        """
        if value is None:
            return None
        
        # Determine PII type from field name if not provided
        if not pii_type and field_name:
            pii_type = self.field_to_type.get(field_name.lower())
        
        # If not PII, return as-is
        if not pii_type or pii_type not in self.redact_types:
            return value
        
        # Convert to string for processing
        value_str = str(value)
        
        # Apply redaction strategy
        if self.redaction_strategy == RedactionStrategy.REMOVE:
            return None
        
        elif self.redaction_strategy == RedactionStrategy.MASK:
            if pii_type == PIIType.EMAIL:
                # Mask email: j***@***.com
                parts = value_str.split('@')
                if len(parts) == 2:
                    local = parts[0]
                    domain = parts[1]
                    masked_local = local[0] + '***' if len(local) > 0 else '***'
                    masked_domain = '***.' + domain.split('.')[-1] if '.' in domain else '***'
                    return f"{masked_local}@{masked_domain}"
                return "***@***"
            elif pii_type == PIIType.PHONE:
                # Mask phone: ***-***-1234
                digits = re.sub(r'\D', '', value_str)
                if len(digits) >= 4:
                    return f"***-***-{digits[-4:]}"
                return "***-***-****"
            elif pii_type == PIIType.SSN:
                return "***-**-****"
            elif pii_type == PIIType.CREDIT_CARD:
                return "****-****-****-****"
            elif pii_type == PIIType.IP_ADDRESS:
                return "***.***.***.***"
            else:
                # Generic masking
                if len(value_str) <= 3:
                    return "***"
                return "***" * min(3, len(value_str) // 3)
        
        elif self.redaction_strategy == RedactionStrategy.PARTIAL_MASK:
            if pii_type == PIIType.EMAIL:
                # Partial mask: j***@example.com
                parts = value_str.split('@')
                if len(parts) == 2:
                    local = parts[0]
                    masked_local = local[0] + '***' if len(local) > 0 else '***'
                    return f"{masked_local}@{parts[1]}"
                return "***@***"
            elif pii_type == PIIType.PHONE:
                digits = re.sub(r'\D', '', value_str)
                if len(digits) >= 4:
                    return f"***-***-{digits[-4:]}"
                return "***-***-****"
            else:
                # Show first and last characters
                if len(value_str) <= 4:
                    return "***"
                return f"{value_str[0]}***{value_str[-1]}"
        
        elif self.redaction_strategy == RedactionStrategy.HASH:
            # Hash the value
            hash_obj = hashlib.sha256(value_str.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()[:16]  # First 16 chars
            return f"sha256:{hash_hex}"
        
        # Fallback: return as-is
        return value
    
    def redact_dict(
        self,
        data: Dict[str, Any],
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Redact PII from a dictionary.
        
        Args:
            data: Dictionary to redact
            recursive: Whether to recursively process nested structures
            
        Returns:
            Dictionary with PII redacted
        """
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        redaction_count = 0
        
        for key, value in data.items():
            # Check if field name indicates PII
            pii_type = self.field_to_type.get(key.lower())
            
            if pii_type and pii_type in self.redact_types:
                # Redact this field
                redacted_value = self.redact_value(value, pii_type, key)
                if redacted_value != value:
                    redaction_count += 1
                    if self.enable_logging:
                        logger.debug(
                            f"Redacted PII field: {key} (type: {pii_type.value})",
                            extra={"field": key, "pii_type": pii_type.value}
                        )
                redacted[key] = redacted_value
            elif recursive and isinstance(value, (dict, list)):
                # Recursively process nested structures
                redacted[key] = self.redact_data(value, recursive=recursive)
            else:
                # Check if value contains PII patterns
                if isinstance(value, str):
                    detected = self.detect_pii(value)
                    if detected:
                        redacted_value = self.redact_value(value, detected[0], key)
                        if redacted_value != value:
                            redaction_count += 1
                            if self.enable_logging:
                                logger.debug(
                                    f"Redacted PII in value: {key}",
                                    extra={"field": key, "pii_types": [d.value for d in detected]}
                                )
                        redacted[key] = redacted_value
                    else:
                        redacted[key] = value
                else:
                    redacted[key] = value
        
        if redaction_count > 0 and self.enable_logging:
            logger.info(
                f"Redacted {redaction_count} PII field(s)",
                extra={"redaction_count": redaction_count}
            )
        
        return redacted
    
    def redact_list(
        self,
        data: List[Any],
        recursive: bool = True
    ) -> List[Any]:
        """
        Redact PII from a list.
        
        Args:
            data: List to redact
            recursive: Whether to recursively process nested structures
            
        Returns:
            List with PII redacted
        """
        if not isinstance(data, list):
            return data
        
        return [
            self.redact_data(item, recursive=recursive)
            for item in data
        ]
    
    def redact_data(
        self,
        data: Any,
        recursive: bool = True
    ) -> Any:
        """
        Redact PII from any data structure.
        
        Args:
            data: Data to redact (dict, list, str, or other)
            recursive: Whether to recursively process nested structures
            
        Returns:
            Data with PII redacted
        """
        if isinstance(data, dict):
            return self.redact_dict(data, recursive=recursive)
        elif isinstance(data, list):
            return self.redact_list(data, recursive=recursive)
        elif isinstance(data, str):
            # Check for PII patterns in string
            detected = self.detect_pii(data)
            if detected:
                return self.redact_value(data, detected[0])
            return data
        else:
            # Non-string primitive types
            return data
    
    def redact_json(
        self,
        json_str: str,
        recursive: bool = True
    ) -> str:
        """
        Redact PII from a JSON string.
        
        Args:
            json_str: JSON string to redact
            recursive: Whether to recursively process nested structures
            
        Returns:
            JSON string with PII redacted
        """
        try:
            data = json.loads(json_str)
            redacted = self.redact_data(data, recursive=recursive)
            return json.dumps(redacted, default=str)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, treat as plain string
            return self.redact_data(json_str, recursive=recursive)


# Global redactor instance (can be configured via settings)
_default_redactor: Optional[PIIRedactor] = None


def get_redactor() -> PIIRedactor:
    """Get the default PII redactor instance."""
    global _default_redactor
    if _default_redactor is None:
        from src.utils.config import settings
        strategy = RedactionStrategy(settings.pii_redaction_strategy)
        _default_redactor = PIIRedactor(
            redaction_strategy=strategy,
            enable_logging=settings.enable_pii_redaction_logging
        )
    return _default_redactor


def redact_pii(data: Any, recursive: bool = True) -> Any:
    """
    Convenience function to redact PII from data.
    
    Args:
        data: Data to redact
        recursive: Whether to recursively process nested structures
        
    Returns:
        Data with PII redacted
    """
    redactor = get_redactor()
    return redactor.redact_data(data, recursive=recursive)
