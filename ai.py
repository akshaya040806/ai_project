import tkinter as tk
from tkinter import messagebox, scrolledtext
import imaplib
import email
from email.header import decode_header
import re
import joblib
import pandas as pd
import ipaddress

# ===============================
# LOAD MODEL + SCALER
# ===============================

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get correct feature names from training
expected_features = scaler.feature_names_in_

# ===============================
# URL FEATURE EXTRACTION
# ===============================

def extract_features(url):
    features = {}

    features["length_url"] = len(url)
    features["nb_dots"] = url.count(".")
    features["nb_hyphens"] = url.count("-")
    features["nb_at"] = url.count("@")
    features["nb_qm"] = url.count("?")
    features["nb_and"] = url.count("&")
    features["nb_eq"] = url.count("=")
    features["nb_slash"] = url.count("/")
    features["nb_www"] = 1 if "www" in url else 0
    features["nb_com"] = 1 if ".com" in url else 0
    features["https_token"] = 1 if "https" in url else 0

    # IP detection
    try:
        hostname = url.split("//")[-1].split("/")[0]
        ipaddress.ip_address(hostname)
        features["ip"] = 1
    except:
        features["ip"] = 0

    hostname = url.split("//")[-1].split("/")[0]
    features["nb_subdomains"] = hostname.count(".")

    return features

# ===============================
# URL EXTRACTOR
# ===============================

def extract_urls(text):
    pattern = r'https?://[^\s]+'
    return re.findall(pattern, text)

# ===============================
# RULE-BASED DETECTION
# ===============================

def rule_based_detection(body, url):
    body_lower = body.lower()
    url_lower = url.lower()

    suspicious_keywords = [
        "suspended",
        "verify",
        "urgent",
        "account termination",
        "confirm identity",
        "action required",
        "limited time",
        "2 hours",
        "security alert"
    ]

    suspicious_tlds = [".xyz", ".top", ".ru", ".tk", ".cf"]

    # Keyword detection
    if any(word in body_lower for word in suspicious_keywords):
        return True

    # Suspicious TLD
    if any(url_lower.endswith(tld) for tld in suspicious_tlds):
        return True

    # Brand mismatch
    if "microsoft" in body_lower and "microsoft.com" not in url_lower:
        return True

    if "paypal" in body_lower and "paypal.com" not in url_lower:
        return True

    if "google" in body_lower and "google.com" not in url_lower:
        return True

    return False

# ===============================
# EMAIL SCANNER FUNCTION
# ===============================

def scan_emails():
    user_email = email_entry.get()
    password = password_entry.get()

    if not user_email or not password:
        messagebox.showerror("Error", "Please enter email and app password.")
        return

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(user_email, password)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()

        result_box.delete("1.0", tk.END)

        for email_id in email_ids[-10:]:  # scan last 10 emails
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")

            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            urls = extract_urls(body)

            phishing_found = False

            for url in urls:
                # Rule-based check first
                if rule_based_detection(body, url):
                    phishing_found = True
                    break

                # ML check
                features = extract_features(url)
                df = pd.DataFrame([features])

                # Add missing expected features with 0
                for col in expected_features:
                    if col not in df.columns:
                        df[col] = 0

                # Keep exact order
                df = df[expected_features]

                X_scaled = scaler.transform(df)
                prediction = model.predict(X_scaled)[0]

                if prediction == 1:
                    phishing_found = True
                    break

            if phishing_found:
                result_box.insert(tk.END, f"⚠ PHISHING DETECTED → {subject}\n")
            else:
                result_box.insert(tk.END, f"SAFE → {subject}\n")

        mail.logout()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ===============================
# GUI
# ===============================

root = tk.Tk()
root.title("Live Email Phishing Detector (Hybrid)")
root.geometry("750x550")

tk.Label(root, text="Email:", font=("Arial", 11)).pack(pady=5)
email_entry = tk.Entry(root, width=50)
email_entry.pack()

tk.Label(root, text="App Password:", font=("Arial", 11)).pack(pady=5)
password_entry = tk.Entry(root, show="*", width=50)
password_entry.pack()

scan_button = tk.Button(root, text="Scan Inbox", command=scan_emails, bg="red", fg="white")
scan_button.pack(pady=10)

result_box = scrolledtext.ScrolledText(root, width=90, height=20)
result_box.pack(pady=10)

root.mainloop()
