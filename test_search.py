# test_search.py — run this from your shl-recommender folder
from retrieval.search import search, filter_by

# Test 1: Java developer query
print("=== Java Developer ===")
results = search("Java developer mid level programming")
for r in results[:5]:
    print(f"  {r['name']} | type={r['test_type']} | score={r['_score']:.3f}")

# Test 2: Sales role
print("\n=== Sales Role ===")
results = search("sales executive persuasion communication")
for r in results[:5]:
    print(f"  {r['name']} | type={r['test_type']} | score={r['_score']:.3f}")

# Test 3: Filter test
print("\n=== Knowledge tests only, Mid-Professional ===")
results = search("software engineer backend")
filtered = filter_by(results, test_types=["K"], job_level="Mid-Professional")
for r in filtered[:5]:
    print(f"  {r['name']} | type={r['test_type']}")