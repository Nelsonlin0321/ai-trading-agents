import os
import smtplib
from loguru import logger
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from src.utils import retry


@retry(retries=5, silence_error=True)
def send_email_gmail(subject: str, recipient: str, html_body: str):
    """
    Send an email using Gmail SMTP server.

    Args:
        subject (str): Email subject
        body (str): Email body in markdown format
        email (str): Recipient email address

    Returns:
        str: Message ID of the sent email
    """

    # Create message
    msg = MIMEMultipart()
    sender = os.getenv("EMAIL")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender:
        logger.warning("EMAIL environment variable to send email is not set")
        return

    if not gmail_password:
        logger.warning(
            "GMAIL_APP_PASSWORD environment variable to send email is not set"
        )
        return

    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach HTML content
    msg.attach(MIMEText(html_body, "html"))

    # Create SMTP session
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Enable TLS

    # Login to Gmail
    server.login(sender, gmail_password)

    # Send email
    server.send_message(msg)

    # Close the connection
    server.quit()

    return f"Email Sent successfully to {recipient}"
