#!/usr/bin/env python3
"""
Test script for the encoder model server.
Tests basic functionality and performance.
"""

import asyncio
import json
import time
import requests
import concurrent.futures
from typing import List, Dict, Any

def test_basic_functionality():
    """Test basic server functionality."""
    print("Testing basic functionality...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        health_data = response.json()
        print(f"✓ Health check passed: {health_data['status']}")
        print(f"  Models loaded: {health_data['models_loaded']}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        models_data = response.json()
        print(f"✓ Models endpoint working: {list(models_data.keys())}")
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        return False
    
    # Test single prediction
    try:
        response = requests.post(f"{base_url}/predict", json={
            "texts": "This is a test sentence to check if the model works correctly."
        })
        response.raise_for_status()
        result = response.json()
        print(f"✓ Single prediction works")
        print(f"  Request ID: {result['request_id']}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Model used: {result['model_used']}")
    except Exception as e:
        print(f"✗ Single prediction failed: {e}")
        return False
    
    # Test batch prediction
    try:
        test_texts = [
            "This is the first test sentence.",
            "Here's another sentence for testing.",
            "And one more sentence to complete the batch."
        ]
        response = requests.post(f"{base_url}/predict", json={
            "texts": test_texts
        })
        response.raise_for_status()
        result = response.json()
        print(f"✓ Batch prediction works")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Batch size: {len(test_texts)}")
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False
    
    return True

def test_performance(num_requests: int = 100, concurrent_requests: int = 10):
    """Test server performance with concurrent requests."""
    print(f"\nTesting performance with {num_requests} requests, {concurrent_requests} concurrent...")
    
    base_url = "http://localhost:8000"
    
    # Generate test sentences
    test_sentences = [
        f"This is test sentence number {i} for performance testing. "
        f"It contains some sample text to evaluate the model performance "
        f"under load with various inputs and different sentence lengths."
        for i in range(num_requests)
    ]
    
    def make_request(text: str) -> Dict[str, Any]:
        """Make a single prediction request."""
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", json={"texts": text})
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            result['client_latency'] = end_time - start_time
            return result
        except Exception as e:
            return {"error": str(e)}
    
    # Run concurrent requests
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        future_to_text = {
            executor.submit(make_request, text): text 
            for text in test_sentences
        }
        
        for future in concurrent.futures.as_completed(future_to_text):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful_requests = [r for r in results if 'error' not in r]
    failed_requests = [r for r in results if 'error' in r]
    
    if failed_requests:
        print(f"✗ {len(failed_requests)} requests failed")
        for req in failed_requests[:5]:  # Show first 5 errors
            print(f"  Error: {req['error']}")
    
    if successful_requests:
        processing_times = [r['processing_time'] for r in successful_requests]
        client_latencies = [r['client_latency'] for r in successful_requests]
        
        print(f"✓ Performance test completed")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {len(successful_requests)}")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(successful_requests) / total_time:.1f} req/s")
        print(f"  Avg processing time: {sum(processing_times) / len(processing_times):.3f}s")
        print(f"  Avg client latency: {sum(client_latencies) / len(client_latencies):.3f}s")
        print(f"  Min latency: {min(client_latencies):.3f}s")
        print(f"  Max latency: {max(client_latencies):.3f}s")
        
        # Calculate percentiles
        sorted_latencies = sorted(client_latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        print(f"  P50 latency: {p50:.3f}s")
        print(f"  P95 latency: {p95:.3f}s") 
        print(f"  P99 latency: {p99:.3f}s")
        
        return True
    
    return False

def test_batch_efficiency():
    """Test that batching actually improves performance."""
    print("\nTesting batch efficiency...")
    
    base_url = "http://localhost:8000"
    test_texts = [
        f"Batch efficiency test sentence {i} with various content lengths and patterns."
        for i in range(20)
    ]
    
    # Test individual requests
    print("Testing individual requests...")
    individual_start = time.time()
    for text in test_texts:
        response = requests.post(f"{base_url}/predict", json={"texts": text})
        response.raise_for_status()
    individual_time = time.time() - individual_start
    
    # Test batch request
    print("Testing batch request...")
    batch_start = time.time()
    response = requests.post(f"{base_url}/predict", json={"texts": test_texts})
    response.raise_for_status()
    batch_time = time.time() - batch_start
    
    speedup = individual_time / batch_time if batch_time > 0 else 1
    
    print(f"  Individual requests: {individual_time:.2f}s")
    print(f"  Batch request: {batch_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("✓ Batching provides significant speedup")
        return True
    else:
        print("⚠ Batching speedup is lower than expected")
        return False

def main():
    """Main test function."""
    print("🚀 Starting encoder model server tests...\n")
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!\n")
                break
        except:
            pass
        time.sleep(1)
        if i == max_retries - 1:
            print("✗ Server not responding after 30 seconds")
            return
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_basic_functionality():
        tests_passed += 1
    
    if test_performance(num_requests=50, concurrent_requests=5):
        tests_passed += 1
    
    if test_batch_efficiency():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()