"""Send email reports via SMTP."""

import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from kubera_reporting.exceptions import EmailError


class EmailSender:
    """Sends emails via SMTP."""

    def __init__(self, recipient: str) -> None:
        """Initialize email sender.

        Args:
            recipient: Email address to send to
        """
        self.recipient = recipient
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

    def send_html_email(
        self,
        subject: str,
        html_content: str,
        from_address: str | None = None,
        chart_image: bytes | None = None,
    ) -> None:
        """Send HTML email via SMTP with optional embedded chart image.

        Args:
            subject: Email subject
            html_content: HTML content
            from_address: From address (optional, defaults to SMTP_USERNAME)
            chart_image: Optional PNG chart image bytes to embed via CID

        Raises:
            EmailError: If sending fails
        """
        if not self.smtp_user or not self.smtp_password:
            raise EmailError(
                "SMTP credentials not found. Please set SMTP_USERNAME and SMTP_PASSWORD environment variables."
            )

        try:
            # Create MIME multipart message
            # Use 'related' if we have images to embed, otherwise 'alternative'
            msg_type = "related" if chart_image else "alternative"
            msg = MIMEMultipart(msg_type)
            msg["Subject"] = subject
            msg["To"] = self.recipient
            msg["From"] = from_address or self.smtp_user

            # Create alternative container for text/html
            if chart_image:
                msg_alternative = MIMEMultipart("alternative")
                msg.attach(msg_alternative)
            else:
                msg_alternative = msg

            # Create plain text version (fallback)
            text_content = "This email requires an HTML-capable email client."
            text_part = MIMEText(text_content, "plain", "utf-8")
            html_part = MIMEText(html_content, "html", "utf-8")

            msg_alternative.attach(text_part)
            msg_alternative.attach(html_part)

            # Attach chart image with Content-ID if provided
            if chart_image:
                image = MIMEImage(chart_image, "png")
                image.add_header("Content-ID", "<chart_image>")
                image.add_header("Content-Disposition", "inline", filename="chart.png")
                msg.attach(image)

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

        except Exception as e:
            raise EmailError(f"Failed to send email via SMTP: {e}") from e
