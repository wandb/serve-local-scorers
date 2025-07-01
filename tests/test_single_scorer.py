#!/usr/bin/env python3
"""
Test a single scorer to verify the setup works.
"""

import asyncio
import aiohttp
import subprocess
import time
import sys


async def test_single_scorer(scorer="fluency", port=8107):
    """Test a single scorer."""
    cmd = [
        "uv", "run", "python", "encoder_server.py", 
        "--weave-scorer", scorer, 
        "--port", str(port),
        "--max-batch-size", "4"
    ]
    
    print(f"🧪 Testing {scorer.upper()} Scorer on port {port}")
    print("=" * 50)
    
    server_process = None
    
    try:
        # Start server
        print(f"Starting server...")
        server_process = subprocess.Popen(
            cmd,
            cwd="/Users/morganmcguire/ML/serve-scorers",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for server to start
        print("Waiting for server to start...")
        for i in range(45):  # Wait up to 45 seconds
            if server_process.poll() is not None:
                stdout, _ = server_process.communicate()
                print(f"❌ Server exited early: {stdout[-500:]}")
                return False
            
            if i > 15:  # Start checking after 15 seconds
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://localhost:{port}/health") as response:
                            if response.status == 200:
                                print(f"✅ Server ready!")
                                break
                except:
                    pass
            
            time.sleep(1)
        else:
            print("⚠️ Server timeout - proceeding anyway")
        
        # Test the server
        print("Testing classification...")
        async with aiohttp.ClientSession() as session:
            # Health check
            async with session.get(f"http://localhost:{port}/health") as response:
                print(f"Health check: {response.status}")
            
            # Classification test
            test_text = "This is a well-written and grammatically correct sentence."
            payload = {"text": test_text}
            async with session.post(f"http://localhost:{port}/classify", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    prediction = result["predictions"][0][0]
                    print(f"✅ SUCCESS!")
                    print(f"   Text: \"{test_text}\"")
                    print(f"   Prediction: {prediction['label']} (score: {prediction['score']:.3f})")
                    print(f"   Processing time: {result['processing_time']:.3f}s")
                    return True
                else:
                    print(f"❌ Classification failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    result = asyncio.run(test_single_scorer())
    sys.exit(0 if result else 1)