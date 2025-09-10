import os
import boto3
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

def send_email(to_email: str, subject: str, body: str, cc_emails: Optional[List[str]] = None):
    """
    Send a plain text email using AWS SES.

    Args:
        to_email (str): Primary recipient email.
        subject (str): Subject of the email.
        body (str): Plain text body of the email.
        cc_emails (list[str], optional): List of CC recipient emails.
    """
    try:
        ses = boto3.client(
            'ses',
            region_name=os.getenv("AWS_SES_REGION"),
            aws_access_key_id=os.getenv("AWS_SES_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SES_SECRET_ACCESS_KEY"),
        )

        destination = {"ToAddresses": [to_email]}
        if cc_emails:
            destination["CcAddresses"] = cc_emails

        response = ses.send_email(
            Source=os.getenv("EMAIL_FROM_ADDRESS"),
            Destination=destination,
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )

        return response

    except Exception as e:
        logging.exception(f"Failed to send email to {to_email} (cc={cc_emails}): {e}")
        raise
