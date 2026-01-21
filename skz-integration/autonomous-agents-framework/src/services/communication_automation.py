from typing import Dict, Any, Optional
import os
import json
import time

try:
    import redis  # type: ignore
except Exception:
    redis = None

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

try:
    import smtplib
    from email.mime.text import MIMEText  # type: ignore
except Exception:
    smtplib = None
    MIMEText = None  # type: ignore

try:
    from sendgrid import SendGridAPIClient  # type: ignore
    from sendgrid.helpers.mail import Mail  # type: ignore
except Exception:
    SendGridAPIClient = None  # type: ignore
    Mail = None  # type: ignore

try:
    from twilio.rest import Client as TwilioClient  # type: ignore
except Exception:
    TwilioClient = None  # type: ignore

try:
    from jinja2 import Environment, FileSystemLoader  # type: ignore
except Exception:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore


class TemplateEngine:
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or os.getenv("TEMPLATE_DIR", "")
        self.env = None
        if Environment is not None and FileSystemLoader is not None and self.template_dir:
            self.env = Environment(loader=FileSystemLoader(self.template_dir), autoescape=True)

    def render(self, name: str, context: Dict[str, Any]) -> str:
        if self.env:
            try:
                tpl = self.env.get_template(name)
                return tpl.render(**context)
            except Exception:
                pass
        return str(context)


class EmailService:
    def __init__(self, template_engine: Optional[TemplateEngine] = None):
        self.provider = os.getenv("EMAIL_PROVIDER", "smtp").lower()
        self.template_engine = template_engine or TemplateEngine()

    def send(self, to_email: str, subject: str, template_name_or_body: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.provider == "sendgrid":
            return self._send_sendgrid(to_email, subject, template_name_or_body, context or {})
        return self._send_smtp(to_email, subject, template_name_or_body, context or {})

    def _send_smtp(self, to_email: str, subject: str, template_or_body: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not smtplib or not MIMEText:
            return {"status": "error", "message": "smtplib not available"}
        host = os.getenv("SMTP_HOST", "localhost")
        port = int(os.getenv("SMTP_PORT", "25"))
        user = os.getenv("SMTP_USER")
        password = os.getenv("SMTP_PASS")
        from_email = os.getenv("SMTP_FROM", user or "no-reply@example.com")

        body = template_or_body
        if self.template_engine and self.template_engine.template_dir:
            body = self.template_engine.render(template_or_body, context)

        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        try:
            server = smtplib.SMTP(host, port, timeout=10)
            if os.getenv("SMTP_STARTTLS", "false").lower() == "true":
                server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(from_email, [to_email], msg.as_string())
            server.quit()
            return {"status": "sent", "provider": "smtp"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _send_sendgrid(self, to_email: str, subject: str, template_or_body: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not SendGridAPIClient or not Mail:
            return {"status": "error", "message": "sendgrid not available"}
        api_key = os.getenv("SENDGRID_API_KEY")
        from_email = os.getenv("SENDGRID_FROM", "no-reply@example.com")
        if not api_key:
            return {"status": "error", "message": "missing SENDGRID_API_KEY"}

        body = template_or_body
        if self.template_engine and self.template_engine.template_dir:
            body = self.template_engine.render(template_or_body, context)

        message = Mail(from_email=from_email, to_emails=to_email, subject=subject, html_content=body)
        try:
            sg = SendGridAPIClient(api_key)
            resp = sg.send(message)
            return {"status": "sent", "provider": "sendgrid", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class SMSService:
    def __init__(self):
        self.provider = os.getenv("SMS_PROVIDER", "twilio").lower()

    def send(self, to_number: str, body: str) -> Dict[str, Any]:
        if self.provider == "sns":
            return self._send_sns(to_number, body)
        return self._send_twilio(to_number, body)

    def _send_twilio(self, to_number: str, body: str) -> Dict[str, Any]:
        if not TwilioClient:
            return {"status": "error", "message": "twilio not available"}
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN") or os.getenv("TWILIO_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        if not sid or not token or not from_number:
            return {"status": "error", "message": "missing twilio config"}
        try:
            client = TwilioClient(sid, token)
            msg = client.messages.create(to=to_number, from_=from_number, body=body)
            return {"status": "sent", "provider": "twilio", "sid": msg.sid}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _send_sns(self, to_number: str, body: str) -> Dict[str, Any]:
        if not boto3:
            return {"status": "error", "message": "boto3 not available"}
        try:
            sns = boto3.client("sns", region_name=os.getenv("AWS_REGION"))
            resp = sns.publish(PhoneNumber=to_number, Message=body)
            return {"status": "sent", "provider": "sns", "message_id": resp.get("MessageId")}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class MessageQueue:
    def __init__(self, namespace: str = "comm"):
        self.provider = os.getenv("QUEUE_PROVIDER", "redis").lower()
        self.ns = namespace
        self.client = None
        if self.provider == "redis" and redis:
            try:
                self.client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6379")), db=0)
            except Exception:
                self.client = None

    def enqueue(self, kind: str, payload: Dict[str, Any]) -> bool:
        if self.client and self.provider == "redis":
            try:
                self.client.rpush(f"{self.ns}:{kind}", json.dumps(payload))
                return True
            except Exception:
                return False
        return False

    def process_once(self, email_service: EmailService, sms_service: SMSService) -> Optional[Dict[str, Any]]:
        if not self.client or self.provider != "redis":
            return None
        for kind in ["email", "sms"]:
            try:
                item = self.client.lpop(f"{self.ns}:{kind}")
                if not item:
                    continue
                data = json.loads(item)
                if kind == "email":
                    res = email_service.send(data["to"], data["subject"], data.get("template_or_body", ""), data.get("context", {}))
                else:
                    res = sms_service.send(data["to"], data["body"])
                data["result"] = res
                return data
            except Exception:
                continue
        return None

    def process_loop(self, email_service: EmailService, sms_service: SMSService, interval: float = 1.0):
        while True:
            processed = self.process_once(email_service, sms_service)
            if not processed:
                time.sleep(interval)
