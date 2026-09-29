import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_suite = [
    (1, "What is a qubit?", "en"),
    (2, "Explain quantum entanglement.", "en"),
    (3, "What is machine learning?", "en"),
    (4, "Teach me VQC.", "en"),
    (5, "Give me ideas to improve MediQAI.", "en"),
    (6, "What was our latest VQC accuracy?", "en"),
    (7, "Why might our VQC perform differently from Logistic Regression?", "en"),
    (8, "Explain our confusion matrix.", "en"),
    (9, "Tell me a joke.", "en"),
    (10, "Talk to me in Telugu.", "en")
]

all_passed = True

for idx, query, lang in test_suite:
    payload = json.dumps({"query": query, "language": lang}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/api/conversation/query",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    
    print("=" * 60)
    print(f"TEST {idx}: \"{query}\"")
    print(f"Category: {data.get('category')}")
    print(f"Tools Used: {data.get('tool_calls')}")
    print(f"Visual State: {data.get('visual_state')}")
    print(f"Spoken Text:\n{data.get('spoken_text')}")
    print()

    # Verification conditions:
    if idx in [1, 2, 3, 4, 5, 9, 10] and data.get("category") != "general":
        print(f"FAILED: Test {idx} was expected to be category 'general'")
        all_passed = False
    elif idx in [6, 8] and (data.get("category") != "project" or not data.get("tool_calls")):
        print(f"FAILED: Test {idx} was expected to be category 'project' with tool calls")
        all_passed = False
    elif idx == 7 and (data.get("category") != "hybrid" or not data.get("tool_calls")):
        print(f"FAILED: Test 7 was expected to be category 'hybrid' with tool calls")
        all_passed = False

if all_passed:
    print("ALL 10 CRITICAL TESTS PASSED SUCCESSFULLY!")
else:
    print("SOME TESTS DID NOT MEET EXPECTED CRITERIA.")
