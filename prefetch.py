"""
Download every built-in city and variable once and save the copies in data/.
Commit the data/ folder so the live app keeps working if NASA POWER is slow or down.

Run with:  python prefetch.py

AI assistance: drafted with Claude (Anthropic) during the hackathon. See README.md.
"""
import time

import power

for city, (lat, lon) in power.CITIES.items():
    for parameter in power.VARIABLES:
        for attempt in range(3):
            try:
                table, meta = power.fetch(lat, lon, parameter)
                state = "already saved" if meta.get("saved_copy") else "downloaded"
                print(f"{city:10s} {parameter:18s} {len(table):4d} months  {state}")
                if state == "downloaded":
                    time.sleep(1)   # be polite to the API
                break
            except Exception as err:
                print(f"{city:10s} {parameter:18s} attempt {attempt + 1} failed: {err}")
                time.sleep(5)
