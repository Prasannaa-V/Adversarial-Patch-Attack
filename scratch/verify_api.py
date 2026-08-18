"""
scratch/verify_api.py
Performs integration tests on the running backend server REST API:
  1. POST a mock attack alert event to /api/events
  2. GET event history and verify the event was logged
  3. POST a manual gate override unlock request to /api/gate
"""

import requests
import json
import sys

URL = "http://localhost:3000"

def run_tests():
    print("=== InfraEdge-Shield API Verification ===")
    
    # Test 1: POST Attack Alert
    alert_payload = {
        "confidence": 0.9654,
        "frame_id": "sim-test-99",
        "frame_path": "/home/prasannaa/Projects/p1/data/frames/2026-08-17/attack_sim.jpg",
        "timestamp": "2026-08-17T23:30:00.000Z"
    }
    
    print("\n[TEST 1] Posting mock alert to /api/events...")
    try:
        r_post = requests.post(f"{URL}/api/events", json=alert_payload, timeout=5)
        print(f"Status Code: {r_post.status_code}")
        print(f"Response: {r_post.text}")
        if r_post.status_code != 201:
            print("❌ Test 1 Failed: Status code is not 201")
            sys.exit(1)
        res_data = r_post.json()
        event_id = res_data.get("id")
        print(f"✅ Test 1 Passed: Event created with ID {event_id}")
    except Exception as e:
        print(f"❌ Test 1 Failed with exception: {e}")
        sys.exit(1)

    # Test 2: GET Event History
    print("\n[TEST 2] Retrieving event history from /api/events...")
    try:
        r_get = requests.get(f"{URL}/api/events", timeout=5)
        print(f"Status Code: {r_get.status_code}")
        data = r_get.json()
        rows = data.get("rows", [])
        print(f"Found {len(rows)} events in history.")
        
        # Verify the created event exists in history
        match = [row for row in rows if row.get("id") == event_id]
        if not match:
            print("❌ Test 2 Failed: Created event ID not found in history")
            sys.exit(1)
        
        saved_event = match[0]
        print(f"Logged Event Detail: {saved_event}")
        print("✅ Test 2 Passed: Event successfully logged in SQLite")
    except Exception as e:
        print(f"❌ Test 2 Failed with exception: {e}")
        sys.exit(1)

    # Test 3: POST Gate Override Command
    override_payload = {
        "action": "unlock"
    }
    print("\n[TEST 3] Posting gate unlock override to /api/gate...")
    try:
        r_gate = requests.post(f"{URL}/api/gate", json=override_payload, timeout=5)
        print(f"Status Code: {r_gate.status_code}")
        print(f"Response: {r_gate.text}")
        if r_gate.status_code != 200:
            print("❌ Test 3 Failed: Status code is not 200")
            sys.exit(1)
        res_gate = r_gate.json()
        if res_gate.get("gate") != "UNLOCKED":
            print("❌ Test 3 Failed: Gate status did not transition to UNLOCKED")
            sys.exit(1)
        print("✅ Test 3 Passed: Gate manual override action processed")
    except Exception as e:
        print(f"❌ Test 3 Failed with exception: {e}")
        sys.exit(1)

    print("\n=== All Integration Tests Completed Successfully! ===")

if __name__ == "__main__":
    run_tests()
