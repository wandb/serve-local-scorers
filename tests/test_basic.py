#!/usr/bin/env python3
"""
Basic test script for the encoder server.
"""

import asyncio
import aiohttp


async def test_basic():
    """Test basic server functionality."""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("🧪 Testing Basic Server Functionality\n")
        
        # Test 1: Health check
        print("1️⃣  Health Check")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print("   ✅ Server is healthy")
                else:
                    print("   ❌ Server is not healthy")
                    return
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            return
        
        # Test 2: Basic classification
        print("\n2️⃣  Basic Classification")
        try:
            payload = {"text": "This is a well-written sentence."}
            async with session.post(f"{base_url}/classify", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    prediction = result['predictions'][0][0]
                    print(f"   ✅ Success: {prediction['label']} (score: {prediction['score']:.3f})")
                else:
                    print(f"   ❌ Failed with status {response.status}")
        except Exception as e:
            print(f"   ❌ Classification failed: {e}")
        
        print("\n🎯 Basic tests completed!")


if __name__ == "__main__":
    print("Make sure to start the server first:")
    print("uv run python encoder_server.py --weave-scorer fluency")
    print("\nThen run this test script.")
    print("=" * 60)
    
    try:
        asyncio.run(test_basic())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the server is running on http://localhost:8000")