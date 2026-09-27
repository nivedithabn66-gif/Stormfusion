"""Verification script for Frontend & Backend integration."""

import urllib.request
import json
import sys

def test_integration():
    endpoints = [
        ('/health', 'GET'),
        ('/api/health', 'GET'),
        ('/api/cyclones/current', 'GET'),
        ('/api/cyclone/FANI_2019/environment', 'GET'),
        ('/api/forecast/FANI_2019', 'GET'),
        ('/api/track/FANI_2019', 'GET'),
        ('/api/satellite/latest', 'GET'),
        ('/api/explainability/gradcam/FANI_2019', 'GET'),
        ('/api/risk/FANI_2019', 'GET'),
        ('/api/alerts', 'GET'),
        ('/api/analogues/FANI_2019', 'GET'),
        ('/api/social-reports', 'GET'),
        ('/api/v1/model/status', 'GET')
    ]

    print("=" * 70)
    print("INTEGRATION VERIFICATION: Frontend Proxy -> FastAPI Backend")
    print("=" * 70)

    all_passed = True

    for ep, method in endpoints:
        url = f"http://127.0.0.1:5173{ep}"
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            status = resp.getcode()
            body = resp.read().decode()
            data = json.loads(body) if body.startswith("{") or body.startswith("[") else body
            summary = f"{type(data).__name__} ({len(data)} items)" if isinstance(data, (list, dict)) else str(data)[:35]
            print(f"[PASS] {method:4} {ep:<40} => Status: {status} | Data: {summary}")
        except Exception as e:
            print(f"[FAIL] {method:4} {ep:<40} => Error: {e}")
            all_passed = False

    # Test POST /api/chat
    chat_req = urllib.request.Request(
        "http://127.0.0.1:5173/api/chat",
        data=json.dumps({"message": "What is the projected landfall time and intensity?"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        c_resp = urllib.request.urlopen(chat_req, timeout=5)
        c_data = json.loads(c_resp.read().decode())
        content_snippet = c_data.get("content", "")[:50]
        print(f"[PASS] POST {'/api/chat':<40} => Status: {c_resp.getcode()} | Reply: {content_snippet}...")
    except Exception as e:
        print(f"[FAIL] POST {'/api/chat':<40} => Error: {e}")
        all_passed = False

    # Test POST /api/social-report
    soc_req = urllib.request.Request(
        "http://127.0.0.1:5173/api/social-report",
        data=json.dumps({
            "location": "Puri Sea Beach",
            "state": "Odisha",
            "coordinates": [19.8, 85.8],
            "content": "Heavy breakers washing over beach road."
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        s_resp = urllib.request.urlopen(soc_req, timeout=5)
        s_data = json.loads(s_resp.read().decode())
        print(f"[PASS] POST {'/api/social-report':<40} => Status: {s_resp.getcode()} | ID: {s_data.get('id')}")
    except Exception as e:
        print(f"[FAIL] POST {'/api/social-report':<40} => Error: {e}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL 15 INTEGRATION ENDPOINTS VERIFIED & OPERATIONAL")
    else:
        print("SOME ENDPOINTS FAILED")
    print("=" * 70)

    return all_passed

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
