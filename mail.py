import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import getpass
import re

_cached_sender = None


def clean_text(s):
    """Remove non-breaking spaces, invisible Unicode, and trim."""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    return s.strip()


def get_sender_credentials():
    """Prompt for sender email + app password once."""
    global _cached_sender
    if _cached_sender is not None:
        return _cached_sender

    print("\n🔐 Email setup required for attendance mail:")
    sender_email = clean_text(input("Enter your email address: ").strip())
    try:
        sender_password = clean_text(getpass.getpass("Enter your app password (hidden): "))
    except Exception:
        sender_password = clean_text(input("Enter your app password: "))

    _cached_sender = (sender_email, sender_password)
    return _cached_sender


def send_professor_email(recognized_people, receiver_email):
    """
    Send attendance email to professor.
    recognized_people → list of (ID, Name)
    """
    sender_email, sender_password = get_sender_credentials()

    subject = "✅ Attendance: Recognized Students from Uploaded Image"
    body_lines = [
        "Dear Professor,",
        "",
        "Here are the students identified in the latest uploaded image:",
        ""
    ]

    if recognized_people:
        for sid, name in recognized_people:
            safe_name = clean_text(name)
            safe_id = clean_text(sid)
            body_lines.append(f"• {safe_name} (ID: {safe_id})")
    else:
        body_lines.append("No recognized students were found.")

    body_lines += ["", "Best regards,", "Automated Attendance System"]
    body = "\n".join(body_lines)

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print(f"[INFO] ✅ Email sent successfully to {receiver_email}")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False
