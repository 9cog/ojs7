import os
from typing import Any, Dict, Optional

try:
    import sendgrid  # type: ignore
    from sendgrid.helpers.mail import Mail  # type: ignore
except Exception:
    sendgrid = None
    Mail = None

try:
    from twilio.rest import Client as TwilioClient  # type: ignore
except Exception:
    TwilioClient = None


class CommunicationAutomation:
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        twilio_sid: Optional[str] = None,
        twilio_token: Optional[str] = None,
        twilio_from: Optional[str] = None,
    ):
        self.sendgrid_api_key = sendgrid_api_key or os.getenv("SENDGRID_API_KEY")
        self.twilio_sid = twilio_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = twilio_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = twilio_from or os.getenv("TWILIO_FROM_NUMBER")

    def send_email(self, to_email: str, subject: str, content: str, from_email: Optional[str] = None) -> Dict[str, Any]:
        if not self.sendgrid_api_key or not sendgrid or not Mail:
            return {"success": True, "provider": "noop", "to": to_email}
        try:
            sg = sendgrid.SendGridAPIClient(self.sendgrid_api_key)
            message = Mail(from_email=from_email or "no-reply@example.com", to_emails=to_email, subject=subject, html_content=content)
            resp = sg.send(message)
            ok = 200 <= int(resp.status_code) < 300
            return {"success": ok, "provider": "sendgrid", "status": int(resp.status_code)}
        except Exception:
            return {"success": False, "provider": "sendgrid", "error": "send_failed"}

    def send_sms(self, to_number: str, body: str) -> Dict[str, Any]:
        if not self.twilio_sid or not self.twilio_token or not self.twilio_from or not TwilioClient:
            return {"success": True, "provider": "noop", "to": to_number}
        try:
            client = TwilioClient(self.twilio_sid, self.twilio_token)
            msg = client.messages.create(to=to_number, from_=self.twilio_from, body=body)
            return {"success": True, "provider": "twilio", "sid": msg.sid}
        except Exception:
            return {"success": False, "provider": "twilio", "error": "send_failed"}
