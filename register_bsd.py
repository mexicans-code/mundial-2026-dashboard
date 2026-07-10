"""Register on BSD API and get API key."""
import requests
import re
import json

s = requests.Session()

# Get registration page
r = s.get("https://sports.bzzoiro.com/register/", timeout=15)
print("Status:", r.status_code)

# Extract CSRF token
csrf = re.search(r'csrfmiddlewaretoken" value="([^"]+)"', r.text)
if not csrf:
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)

if csrf:
    token = csrf.group(1)
    print("CSRF token found:", token[:30])
else:
    print("No CSRF token found")
    # Try to find any pattern
    print(r.text[:2000])

# Submit registration
data = {
    "csrfmiddlewaretoken": token,
    "username": "apuestas_mx",
    "email": "apuestas_mx@tempmail.com",
    "password1": "Apuestas2026!",
    "password2": "Apuestas2026!",
}
headers = {
    "Referer": "https://sports.bzzoiro.com/register/",
}
r2 = s.post("https://sports.bzzoiro.com/register/", data=data, headers=headers, timeout=15)
print("Registration status:", r2.status_code)
print("URL:", r2.url)

# If redirected to profile, we can get the API key
if r2.url and "profile" in r2.url or "account" in r2.url:
    # Look for API key in the page
    api_key_match = re.search(r'API[^<]{0,30}([A-Za-z0-9]{40,})', r2.text)
    if api_key_match:
        print("API Key found:", api_key_match.group(1))
    else:
        print("Profile page:", r2.url)
        print(r2.text[:2000])
else:
    # Check for errors
    errors = re.findall(r'class="errorlist"[^>]*>(.*?)</ul>', r2.text, re.DOTALL)
    if errors:
        for e in errors:
            print("Error:", e)
    else:
        print(r2.text[:2000])
