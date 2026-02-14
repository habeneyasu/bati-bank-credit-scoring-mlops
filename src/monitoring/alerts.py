"""
Alerting and Notification System

Provides real-time alerting for:
- Model drift detection
- SLA violations (latency > 200ms)
- Error rate spikes
- Model prediction anomalies
- System health issues

Supports multiple notification channels:
- Email
- Slack webhooks
- Webhooks (generic)
- Database logging
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from src.utils.logging import get_logger
from src.utils.config import settings
from src.database.connection import get_db_session
from src.database.models import AuditLog

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DATABASE = "database"
    LOG = "log"


class Alert:
    """Alert model."""
    
    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        alert_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            alert_type: Type of alert (drift, sla, error, etc.)
            metadata: Additional alert metadata
        """
        self.title = title
        self.message = message
        self.severity = severity
        self.alert_type = alert_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        self.acknowledged = False
        self.acknowledged_at = None
        self.acknowledged_by = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by
        }


class AlertManager:
    """
    Manages alerting and notifications.
    """
    
    def __init__(
        self,
        enabled_channels: Optional[List[AlertChannel]] = None,
        email_config: Optional[Dict[str, str]] = None,
        slack_webhook_url: Optional[str] = None,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            enabled_channels: List of enabled notification channels
            email_config: Email configuration (smtp_host, smtp_port, from_email, password)
            slack_webhook_url: Slack webhook URL
            webhook_url: Generic webhook URL
        """
        self.enabled_channels = enabled_channels or [AlertChannel.LOG, AlertChannel.DATABASE]
        self.email_config = email_config or {}
        self.slack_webhook_url = slack_webhook_url
        self.webhook_url = webhook_url
        self.logger = get_logger(__name__)
        
        # Alert history (in-memory, could be moved to database)
        self.alert_history: List[Alert] = []
        self.max_history = 1000
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ) -> Dict[str, bool]:
        """
        Send alert through specified channels.
        
        Args:
            alert: Alert to send
            channels: Channels to use (defaults to enabled_channels)
            
        Returns:
            Dictionary mapping channel to success status
        """
        channels = channels or self.enabled_channels
        results = {}
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Send through each channel
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    results[channel.value] = self._send_email(alert)
                elif channel == AlertChannel.SLACK:
                    results[channel.value] = self._send_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    results[channel.value] = self._send_webhook(alert)
                elif channel == AlertChannel.DATABASE:
                    results[channel.value] = self._log_to_database(alert)
                elif channel == AlertChannel.LOG:
                    results[channel.value] = self._log_alert(alert)
                else:
                    results[channel.value] = False
            except Exception as e:
                self.logger.error(f"Error sending alert via {channel.value}: {e}", exc_info=True)
                results[channel.value] = False
        
        return results
    
    def _send_email(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            if not self.email_config.get("smtp_host"):
                self.logger.warning("Email not configured, skipping email alert")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get("from_email", "alerts@batibank.com")
            msg['To'] = self.email_config.get("to_email", "mlops-team@batibank.com")
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value.upper()}
            Time: {alert.timestamp.isoformat()}
            Type: {alert.alert_type}
            
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(
                self.email_config.get("smtp_host", "smtp.gmail.com"),
                self.email_config.get("smtp_port", 587)
            )
            server.starttls()
            
            if self.email_config.get("password"):
                server.login(
                    self.email_config.get("from_email"),
                    self.email_config.get("password")
                )
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}", exc_info=True)
            return False
    
    def _send_slack(self, alert: Alert) -> bool:
        """Send alert via Slack webhook."""
        try:
            if not self.slack_webhook_url:
                self.logger.warning("Slack webhook not configured, skipping Slack alert")
                return False
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": alert.alert_type,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": False
                            }
                        ],
                        "footer": "MLOps Credit Scoring Platform",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            if alert.metadata:
                payload["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": json.dumps(alert.metadata, indent=2),
                    "short": False
                })
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}", exc_info=True)
            return False
    
    def _send_webhook(self, alert: Alert) -> bool:
        """Send alert via generic webhook."""
        try:
            if not self.webhook_url:
                self.logger.warning("Webhook URL not configured, skipping webhook alert")
                return False
            
            payload = alert.to_dict()
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            return response.status_code in [200, 201, 204]
            
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}", exc_info=True)
            return False
    
    def _log_to_database(self, alert: Alert) -> bool:
        """Log alert to database audit log."""
        try:
            with get_db_session() as session:
                audit_log = AuditLog(
                    username="system",
                    action=f"alert_{alert.alert_type}",
                    resource_type="alert",
                    resource_id=alert.alert_type,
                    status_code=200 if alert.severity != AlertSeverity.CRITICAL else 500,
                    success=alert.severity != AlertSeverity.CRITICAL,
                    error_message=alert.message if alert.severity == AlertSeverity.CRITICAL else None,
                    log_metadata={
                        "alert_title": alert.title,
                        "alert_severity": alert.severity.value,
                        "alert_metadata": alert.metadata
                    }
                )
                
                session.add(audit_log)
                session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging alert to database: {e}", exc_info=True)
            return False
    
    def _log_alert(self, alert: Alert) -> bool:
        """Log alert to application logs."""
        try:
            log_level = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical"
            }.get(alert.severity, "info")
            
            getattr(self.logger, log_level)(
                f"ALERT: {alert.title} - {alert.message}",
                extra={
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "metadata": alert.metadata
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}", exc_info=True)
            return False
    
    def check_sla_violation(
        self,
        p95_latency_ms: float,
        threshold_ms: float = 200.0
    ) -> Optional[Alert]:
        """
        Check for SLA violation and create alert if needed.
        
        Args:
            p95_latency_ms: 95th percentile latency in milliseconds
            threshold_ms: SLA threshold in milliseconds
            
        Returns:
            Alert if violation detected, None otherwise
        """
        if p95_latency_ms > threshold_ms:
            return Alert(
                title="SLA Violation: High Latency",
                message=f"P95 latency ({p95_latency_ms:.2f}ms) exceeds threshold ({threshold_ms}ms)",
                severity=AlertSeverity.WARNING if p95_latency_ms < threshold_ms * 1.5 else AlertSeverity.ERROR,
                alert_type="sla_violation",
                metadata={
                    "p95_latency_ms": p95_latency_ms,
                    "threshold_ms": threshold_ms,
                    "violation_percent": ((p95_latency_ms - threshold_ms) / threshold_ms) * 100
                }
            )
        return None
    
    def check_error_rate_spike(
        self,
        error_rate: float,
        threshold: float = 0.05
    ) -> Optional[Alert]:
        """
        Check for error rate spike and create alert if needed.
        
        Args:
            error_rate: Current error rate (0-1)
            threshold: Error rate threshold
            
        Returns:
            Alert if spike detected, None otherwise
        """
        if error_rate > threshold:
            return Alert(
                title="Error Rate Spike Detected",
                message=f"Error rate ({error_rate*100:.2f}%) exceeds threshold ({threshold*100:.2f}%)",
                severity=AlertSeverity.ERROR if error_rate > threshold * 2 else AlertSeverity.WARNING,
                alert_type="error_rate_spike",
                metadata={
                    "error_rate": error_rate,
                    "threshold": threshold,
                    "error_rate_percent": error_rate * 100
                }
            )
        return None
    
    def check_drift_alert(
        self,
        feature_name: str,
        psi: float,
        drift_severity: str
    ) -> Optional[Alert]:
        """
        Create alert for detected drift.
        
        Args:
            feature_name: Feature name with drift
            psi: PSI value
            drift_severity: Severity level
            
        Returns:
            Alert for drift detection
        """
        severity_map = {
            "major": AlertSeverity.ERROR,
            "minor": AlertSeverity.WARNING,
            "none": AlertSeverity.INFO
        }
        
        if drift_severity == "none":
            return None
        
        return Alert(
            title=f"Data Drift Detected: {feature_name}",
            message=f"Feature '{feature_name}' shows {drift_severity} drift (PSI: {psi:.4f})",
            severity=severity_map.get(drift_severity, AlertSeverity.WARNING),
            alert_type="drift_detection",
            metadata={
                "feature_name": feature_name,
                "psi": psi,
                "drift_severity": drift_severity
            }
        )


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
