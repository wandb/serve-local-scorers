#!/usr/bin/env python3
"""
Test script for Weave Scorer Server
Tests all available endpoints with sample data.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Test server configuration
SERVER_URL = "http://localhost:8001"

# Test data
TEST_TEXTS = {
    "neutral": "The weather is nice today.",
    "potentially_biased": "Women are not good at math and science.",
    "potentially_toxic": "I hate everyone and everything.",
    "coherent": "First we go to the store. Then we buy groceries. Finally we come home.",
    "incoherent": "Store banana purple elephant mathematics seventeen.",
    "fluent": "This is a well-written sentence with proper grammar and structure.",
    "disfluent": "This sentence are not good written and have bad grammar structure.",
    "context_example": "The capital of France is Paris.",
    "query_example": "What is the capital of France?",
    "hallucinated": "The capital of France is London.",
    "pii_text": "My name is John Doe and my email is john.doe@example.com"
}

class WeaveServerTester:
    """Test client for Weave Scorer Server."""
    
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> Dict[str, Any]:
        """Test health endpoint."""
        print("🏥 Testing health endpoint...")
        async with self.session.get(f"{self.base_url}/health") as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            print(f"   Response: {result}")
            return result
    
    async def test_scorers_list(self) -> Dict[str, Any]:
        """Test scorers list endpoint."""
        print("📋 Testing scorers list endpoint...")
        async with self.session.get(f"{self.base_url}/scorers") as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            print(f"   Available scorers: {result.get('loaded_count', 0)}/{result.get('total_count', 0)}")
            return result
    
    async def test_bias_scorer(self) -> Dict[str, Any]:
        """Test bias scorer."""
        print("⚖️  Testing bias scorer...")
        data = {
            "text": TEST_TEXTS["potentially_biased"],
            "scorer_type": "bias"
        }
        async with self.session.post(f"{self.base_url}/score/bias", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_toxicity_scorer(self) -> Dict[str, Any]:
        """Test toxicity scorer."""
        print("☠️  Testing toxicity scorer...")
        data = {
            "text": TEST_TEXTS["potentially_toxic"],
            "scorer_type": "toxicity"
        }
        async with self.session.post(f"{self.base_url}/score/toxicity", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_coherence_scorer(self) -> Dict[str, Any]:
        """Test coherence scorer."""
        print("🧩 Testing coherence scorer...")
        data = {
            "text": TEST_TEXTS["incoherent"],
            "scorer_type": "coherence"
        }
        async with self.session.post(f"{self.base_url}/score/coherence", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_fluency_scorer(self) -> Dict[str, Any]:
        """Test fluency scorer."""
        print("✍️  Testing fluency scorer...")
        data = {
            "text": TEST_TEXTS["disfluent"],
            "scorer_type": "fluency"
        }
        async with self.session.post(f"{self.base_url}/score/fluency", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_context_relevance_scorer(self) -> Dict[str, Any]:
        """Test context relevance scorer."""
        print("🎯 Testing context relevance scorer...")
        data = {
            "text": TEST_TEXTS["context_example"],
            "query": TEST_TEXTS["query_example"],
            "context": TEST_TEXTS["context_example"],
            "scorer_type": "context_relevance"
        }
        async with self.session.post(f"{self.base_url}/score/context-relevance", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_hallucination_scorer(self) -> Dict[str, Any]:
        """Test hallucination scorer."""
        print("👻 Testing hallucination scorer...")
        data = {
            "text": TEST_TEXTS["hallucinated"],
            "context": TEST_TEXTS["context_example"],
            "scorer_type": "hallucination"
        }
        async with self.session.post(f"{self.base_url}/score/hallucination", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_pii_scorer(self) -> Dict[str, Any]:
        """Test PII scorer."""
        print("🔒 Testing PII scorer...")
        data = {
            "text": TEST_TEXTS["pii_text"],
            "scorer_type": "pii",
            "language": "en"
        }
        async with self.session.post(f"{self.base_url}/score/pii", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_trust_scorer(self) -> Dict[str, Any]:
        """Test trust scorer."""
        print("🤝 Testing trust scorer...")
        data = {
            "text": TEST_TEXTS["context_example"],
            "query": TEST_TEXTS["query_example"],
            "context": TEST_TEXTS["context_example"],
            "scorer_type": "trust"
        }
        async with self.session.post(f"{self.base_url}/score/trust", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_batch_scoring(self) -> Dict[str, Any]:
        """Test batch scoring with multiple texts."""
        print("📦 Testing batch scoring...")
        data = {
            "text": [TEST_TEXTS["neutral"], TEST_TEXTS["potentially_biased"]],
            "scorer_type": "bias"
        }
        async with self.session.post(f"{self.base_url}/score/bias", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Batch size: {result['batch_size']}")
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {len(result['scores'])} results")
            else:
                print(f"   Error: {result}")
            return result
    
    async def test_generic_endpoint(self) -> Dict[str, Any]:
        """Test generic scoring endpoint."""
        print("🎪 Testing generic scoring endpoint...")
        data = {
            "text": TEST_TEXTS["neutral"],
            "scorer_type": "fluency"
        }
        async with self.session.post(f"{self.base_url}/score", json=data) as resp:
            result = await resp.json()
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                print(f"   Processing time: {result['processing_time']:.3f}s")
                print(f"   Scores: {result['scores']}")
            else:
                print(f"   Error: {result}")
            return result
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Weave Scorer Server Tests")
        print("=" * 50)
        
        try:
            # Basic tests
            await self.test_health()
            print()
            
            await self.test_scorers_list()
            print()
            
            # Individual scorer tests
            test_methods = [
                self.test_bias_scorer,
                self.test_toxicity_scorer,
                self.test_coherence_scorer,
                self.test_fluency_scorer,
                self.test_context_relevance_scorer,
                self.test_hallucination_scorer,
                self.test_pii_scorer,
                self.test_trust_scorer,
            ]
            
            for test_method in test_methods:
                try:
                    await test_method()
                except Exception as e:
                    print(f"   ❌ Test failed: {e}")
                print()
            
            # Advanced tests
            await self.test_batch_scoring()
            print()
            
            await self.test_generic_endpoint()
            print()
            
            print("✅ All tests completed!")
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")

def print_usage():
    """Print usage instructions."""
    print("""
Weave Scorer Server Test Script
==============================

Usage:
  python test_weave_server.py

Before running tests:
1. Start the Weave scorer server:
   python weave_scorer_server.py

2. Make sure you have weave[scorers] installed:
   pip install weave[scorers]

3. Run the tests:
   python test_weave_server.py

The test script will verify all endpoints and scorers are working correctly.
    """)

async def main():
    """Main test function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage()
        return
    
    print("🧪 Weave Scorer Server Test Suite")
    print(f"📡 Testing server at: {SERVER_URL}")
    print()
    
    # Test server connectivity first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/health", timeout=5) as resp:
                if resp.status != 200:
                    print(f"❌ Server not responding correctly (status: {resp.status})")
                    print("Make sure the server is running with: python weave_scorer_server.py")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running with: python weave_scorer_server.py")
        return
    
    # Run all tests
    async with WeaveServerTester() as tester:
        await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())