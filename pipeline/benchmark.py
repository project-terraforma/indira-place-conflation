import time
from groq import Groq

MODELS = [
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

TEST_PROMPT = """
Decide if these two POIs refer to the same place.
Record A: 100 main street, springfield
Record B: 101 main street, springfield
Return JSON: {"same_place":0,"confidence":0.0,"explanation":"x"}
"""

client = Groq()

def safe_get_tokens(resp):
    """Return completion tokens if available, else None."""
    try:
        if hasattr(resp, "usage"):
            usage = resp.usage
            if hasattr(usage, "completion_tokens"):
                return usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                return usage.total_tokens
    except:
        pass
    return None


def benchmark_model(model_name, runs=5):
    print(f"\n=== Testing {model_name} ===")

    latencies = []
    token_counts = []

    for i in range(runs):
        start = time.time()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            temperature=0.0
        )
        end = time.time()

        latency = end - start
        latencies.append(latency)

        tokens = safe_get_tokens(resp)
        token_counts.append(tokens)

        print(f"Run {i+1}: {latency:.3f}s, tokens={tokens}")

    avg_lat = sum(latencies) / runs

    valid_tokens = [t for t in token_counts if t is not None]
    avg_tokens = sum(valid_tokens) / len(valid_tokens) if valid_tokens else None

    print(f"\n📌 Results for {model_name}:")
    print(f"  Average latency: {avg_lat:.3f} sec")
    print(f"  Average tokens: {avg_tokens if avg_tokens else 'N/A'}")
    print(f"  Throughput: {1/avg_lat:.2f} matches/sec")

    return {
        "model": model_name,
        "avg_latency": avg_lat,
        "avg_tokens": avg_tokens,
        "throughput": 1/avg_lat
    }


if __name__ == "__main__":
    results = []

    for m in MODELS:
        try:
            results.append(benchmark_model(m))
        except Exception as e:
            print(f"Error testing {m}: {e}")

    print("\n\n=== FINAL SUMMARY ===")
    for r in results:
        print(
            f"{r['model']} — "
            f"{r['avg_latency']:.3f}s avg, "
            f"tokens={r['avg_tokens'] if r['avg_tokens'] else 'N/A'}, "
            f"{r['throughput']:.2f} matches/sec"
        )
