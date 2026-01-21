import json
import sys
from datetime import datetime

import requests


SERVICES = {
    "api_gateway": "http://localhost:5000/health",
    "research_agent": "http://localhost:5001/health",
    "submission_agent": "http://localhost:5002/health",
    "editorial_agent": "http://localhost:5003/health",
    "review_agent": "http://localhost:5004/health",
    "quality_agent": "http://localhost:5005/health",
    "publishing_agent": "http://localhost:5006/health",
    "analytics_agent": "http://localhost:5007/health",
}


def check(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        status = "healthy" if r.status_code == 200 else "unhealthy"
        return {
            "status": status,
            "code": r.status_code,
            "response_time": r.elapsed.total_seconds(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def main():
    results = {}
    overall_ok = True
    for name, url in SERVICES.items():
        res = check(url)
        results[name] = res
        if res.get("status") != "healthy":
            overall_ok = False
    output = {
        "overall_status": "healthy" if overall_ok else "unhealthy",
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "services": results,
    }
    print(json.dumps(output, indent=2))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
