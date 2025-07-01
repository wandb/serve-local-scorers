#!/usr/bin/env python3
"""
Test script to verify all Weave scorers work correctly with the server.
"""

import asyncio
import aiohttp
import subprocess
import time
import sys
import signal
import os


class ScorerTester:
    """Test all available scorers."""
    
    def __init__(self):
        self.scorers = [
            "bias", "toxicity", "hallucination", 
            "context_relevance", "coherence", "fluency"
        ]
        self.test_texts = {
            "bias": "Men are better at math than women.",
            "toxicity": "You are an idiot and should shut up!",
            "hallucination": "The Eiffel Tower is located in London.",
            "context_relevance": "The sky is blue. What is the capital of France?",
            "coherence": "The cat. Running fast. Blue yesterday tomorrow.",
            "fluency": "This is a well-written and grammatically correct sentence."
        }
        self.server_process = None
    
    def start_server(self, scorer: str, port: int = 8100) -> bool:
        """Start the server with a specific scorer."""
        cmd = [
            "uv", "run", "python", "encoder_server.py", 
            "--weave-scorer", scorer, 
            "--port", str(port),
            "--max-batch-size", "4"  # Smaller batch for testing
        ]
        
        try:
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Capture both stdout and stderr
                text=True,
                cwd="/Users/morganmcguire/ML/serve-scorers"  # Set working directory
            )
            
            # Wait for server to start and check for errors
            print(f"  Starting server with {scorer} scorer on port {port}...")
            
            # Wait and check if server started successfully
            for i in range(30):  # Check for 30 seconds
                if self.server_process.poll() is not None:
                    # Process exited, capture error
                    stdout, _ = self.server_process.communicate()
                    print(f"  ❌ Server exited early. Output: {stdout[-500:]}")  # Last 500 chars
                    return False
                
                time.sleep(1)
                
                # Try to connect to check if server is ready
                if i > 10:  # Start checking after 10 seconds
                    try:
                        import requests
                        response = requests.get(f"http://localhost:{port}/health", timeout=2)
                        if response.status_code == 200:
                            print(f"  ✅ Server ready on port {port}")
                            return True
                    except:
                        continue
            
            print(f"  ⚠️ Server started but not responding on port {port}")
            return True  # Assume it's working even if health check failed
            
        except Exception as e:
            print(f"  ❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the current server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            except Exception:
                pass
            self.server_process = None
    
    async def test_scorer(self, scorer: str, port: int = 8100) -> dict:
        """Test a specific scorer."""
        base_url = f"http://localhost:{port}"
        test_text = self.test_texts[scorer]
        
        try:
            async with aiohttp.ClientSession() as session:
                # Health check
                async with session.get(f"{base_url}/health") as response:
                    if response.status != 200:
                        return {"success": False, "error": "Health check failed"}
                
                # Model info check
                async with session.get(f"{base_url}/info") as response:
                    if response.status != 200:
                        return {"success": False, "error": "Info endpoint failed"}
                    info = await response.json()
                
                # Classification test
                payload = {"text": test_text}
                async with session.post(f"{base_url}/classify", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {"success": False, "error": f"Classification failed: {error_text}"}
                    
                    result = await response.json()
                    
                    if "predictions" not in result or not result["predictions"]:
                        return {"success": False, "error": "No predictions returned"}
                    
                    prediction = result["predictions"][0][0]
                    
                    return {
                        "success": True,
                        "label": prediction["label"],
                        "score": prediction["score"],
                        "processing_time": result.get("processing_time", 0),
                        "model_info": info.get("model_name", "unknown")
                    }
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_all_scorers(self):
        """Test all available scorers."""
        print("🧪 Testing All Weave Scorers\n")
        print("=" * 70)
        
        results = {}
        
        for i, scorer in enumerate(self.scorers, 1):
            port = 8100 + i  # Use different port for each scorer
            print(f"\n{i}️⃣  Testing {scorer.upper()} Scorer")
            print("-" * 50)
            
            # Start server
            if not self.start_server(scorer, port):
                results[scorer] = {"success": False, "error": "Failed to start server"}
                print(f"  ❌ Failed to start server for {scorer}")
                continue
            
            try:
                # Test the scorer
                result = await self.test_scorer(scorer, port)
                results[scorer] = result
                
                if result["success"]:
                    print(f"  ✅ SUCCESS")
                    print(f"     Model: {result['model_info']}")
                    print(f"     Test text: \"{self.test_texts[scorer]}\"")
                    print(f"     Prediction: {result['label']} (score: {result['score']:.3f})")
                    print(f"     Processing time: {result['processing_time']:.3f}s")
                else:
                    print(f"  ❌ FAILED: {result['error']}")
                    
            except Exception as e:
                results[scorer] = {"success": False, "error": str(e)}
                print(f"  ❌ FAILED: {e}")
            
            finally:
                # Stop server
                self.stop_server()
                time.sleep(2)  # Brief pause between tests
        
        # Summary
        self.print_summary(results)
        return results
    
    def print_summary(self, results: dict):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        print(f"Total scorers tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        if successful == total:
            print("\n🎉 ALL SCORERS WORKING PERFECTLY!")
        else:
            print("\n⚠️  Some scorers failed:")
            for scorer, result in results.items():
                if not result["success"]:
                    print(f"  - {scorer}: {result['error']}")


async def main():
    """Main test runner."""
    tester = ScorerTester()
    
    try:
        results = await tester.test_all_scorers()
        
        # Exit with error code if any tests failed
        failed_count = sum(1 for r in results.values() if not r["success"])
        sys.exit(failed_count)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        tester.stop_server()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        tester.stop_server()
        sys.exit(1)


if __name__ == "__main__":
    print("🔍 Comprehensive Weave Scorer Test Suite")
    print("This will test all 6 available scorers by starting the server")
    print("with each model and running classification tests.\n")
    
    asyncio.run(main())