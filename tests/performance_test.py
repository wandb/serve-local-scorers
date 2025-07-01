#!/usr/bin/env python3
"""
Comprehensive performance testing for encoder server.
Tests throughput across different sequence lengths and batch sizes.
"""

import asyncio
import time
import statistics
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
import string


class PerformanceTester:
    """Comprehensive performance tester for encoder server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_text_with_target_tokens(self, target_tokens: int, seed: int = 42) -> str:
        """Generate text with approximately target number of tokens."""
        random.seed(seed)
        
        # Average English word is ~4-5 characters, tokenizers often split into subwords
        # So we estimate ~1.3 tokens per word on average
        target_words = int(target_tokens / 1.3)
        
        # Generate random words of varying lengths
        words = []
        for i in range(target_words):
            word_length = random.randint(3, 12)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        
        # Add some punctuation and structure
        text_parts = []
        current_sentence = []
        
        for i, word in enumerate(words):
            current_sentence.append(word)
            
            # End sentence randomly (every 8-15 words)
            if len(current_sentence) >= random.randint(8, 15) or i == len(words) - 1:
                sentence = ' '.join(current_sentence)
                # Capitalize first letter and add punctuation
                sentence = sentence[0].upper() + sentence[1:] + random.choice(['.', '!', '?'])
                text_parts.append(sentence)
                current_sentence = []
        
        return ' '.join(text_parts)
    
    def generate_test_data(self, sequence_lengths: List[int], samples_per_length: int = 5) -> Dict[int, List[str]]:
        """Generate test data for different sequence lengths."""
        print(f"🔄 Generating test data for sequence lengths: {sequence_lengths}")
        
        test_data = {}
        for seq_len in sequence_lengths:
            texts = []
            for i in range(samples_per_length):
                text = self.generate_text_with_target_tokens(seq_len, seed=seq_len + i)
                texts.append(text)
            test_data[seq_len] = texts
            print(f"  Generated {len(texts)} samples for {seq_len} tokens")
        
        return test_data
    
    async def get_actual_token_count(self, text: str) -> int:
        """Get actual token count from the server (if available)."""
        # For now, estimate based on text length
        # In a real scenario, you'd tokenize using the same tokenizer as the server
        return len(text.split()) * 1.3  # Rough estimate
    
    async def classify_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Classify a batch of texts and return timing info."""
        payload = {"text": texts, "return_all_scores": False}
        
        start_time = time.time()
        async with self.session.post(f"{self.base_url}/classify", json=payload) as response:
            if response.status != 200:
                raise Exception(f"Server error: {response.status}")
            result = await response.json()
        
        end_time = time.time()
        
        return {
            "server_processing_time": result.get("processing_time", 0),
            "total_request_time": end_time - start_time,
            "batch_size": len(texts),
            "predictions": result.get("predictions", [])
        }
    
    async def test_sequence_length_performance(
        self, 
        test_data: Dict[int, List[str]], 
        batch_size: int = 8,
        rounds: int = 3
    ) -> Dict[int, Dict[str, float]]:
        """Test performance across different sequence lengths."""
        print(f"\n📏 Testing sequence length performance (batch_size={batch_size}, rounds={rounds})")
        
        results = {}
        
        for seq_len in sorted(test_data.keys()):
            print(f"\n  Testing sequence length ~{seq_len} tokens:")
            texts = test_data[seq_len]
            
            # Create batches
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                if batch:
                    batches.append(batch)
            
            round_times = []
            total_texts = 0
            server_processing_times = []
            
            for round_num in range(rounds):
                round_start = time.time()
                
                # Process all batches in this round
                tasks = [self.classify_batch(batch) for batch in batches]
                batch_results = await asyncio.gather(*tasks)
                
                round_time = time.time() - round_start
                round_texts = sum(len(batch) for batch in batches)
                total_texts += round_texts
                
                # Collect server processing times
                for result in batch_results:
                    server_processing_times.append(result["server_processing_time"])
                
                throughput = round_texts / round_time
                print(f"    Round {round_num + 1}: {round_time:.3f}s, {throughput:.1f} texts/sec")
                round_times.append(round_time)
            
            # Calculate metrics
            avg_round_time = statistics.mean(round_times)
            avg_server_time = statistics.mean(server_processing_times)
            total_throughput = total_texts / sum(round_times)
            
            results[seq_len] = {
                "avg_round_time": avg_round_time,
                "avg_server_processing_time": avg_server_time,
                "throughput_texts_per_sec": total_throughput,
                "total_texts_processed": total_texts,
                "tokens_per_sec": total_throughput * seq_len
            }
            
            print(f"    Avg throughput: {total_throughput:.1f} texts/sec, {results[seq_len]['tokens_per_sec']:.0f} tokens/sec")
        
        return results
    
    async def test_batch_size_performance(
        self, 
        texts: List[str], 
        batch_sizes: List[int],
        rounds: int = 3
    ) -> Dict[int, Dict[str, float]]:
        """Test performance across different batch sizes."""
        print(f"\n📦 Testing batch size performance with {len(texts)} texts (rounds={rounds})")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n  Testing batch size {batch_size}:")
            
            # Create batches
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                if batch:
                    batches.append(batch)
            
            round_times = []
            total_texts = 0
            server_processing_times = []
            request_latencies = []
            
            for round_num in range(rounds):
                round_start = time.time()
                
                # Process all batches
                batch_results = []
                for batch in batches:
                    result = await self.classify_batch(batch)
                    batch_results.append(result)
                    request_latencies.append(result["total_request_time"])
                
                round_time = time.time() - round_start
                round_texts = sum(len(batch) for batch in batches)
                total_texts += round_texts
                
                # Collect server processing times
                for result in batch_results:
                    server_processing_times.append(result["server_processing_time"])
                
                throughput = round_texts / round_time
                print(f"    Round {round_num + 1}: {round_time:.3f}s, {throughput:.1f} texts/sec")
                round_times.append(round_time)
            
            # Calculate metrics
            results[batch_size] = {
                "avg_round_time": statistics.mean(round_times),
                "avg_server_processing_time": statistics.mean(server_processing_times),
                "avg_request_latency": statistics.mean(request_latencies),
                "p95_request_latency": statistics.quantiles(request_latencies, n=20)[18] if len(request_latencies) > 20 else max(request_latencies),
                "throughput_texts_per_sec": total_texts / sum(round_times),
                "total_texts_processed": total_texts,
                "num_batches": len(batches) * rounds
            }
            
            print(f"    Avg throughput: {results[batch_size]['throughput_texts_per_sec']:.1f} texts/sec")
            print(f"    Avg latency: {results[batch_size]['avg_request_latency']:.3f}s")
        
        return results
    
    async def test_concurrent_load(
        self, 
        texts: List[str], 
        concurrent_requests: List[int],
        batch_size: int = 4
    ) -> Dict[int, Dict[str, float]]:
        """Test performance under concurrent load."""
        print(f"\n🚀 Testing concurrent load performance (batch_size={batch_size})")
        
        results = {}
        
        for concurrency in concurrent_requests:
            print(f"\n  Testing {concurrency} concurrent requests:")
            
            # Create batches for concurrent requests
            batch_texts = texts[:batch_size]
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for _ in range(concurrency):
                task = self.classify_batch(batch_texts)
                tasks.append(task)
            
            # Execute all tasks concurrently
            concurrent_results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            total_texts = len(batch_texts) * concurrency
            
            # Collect metrics
            server_times = [r["server_processing_time"] for r in concurrent_results]
            request_times = [r["total_request_time"] for r in concurrent_results]
            
            results[concurrency] = {
                "total_time": total_time,
                "throughput_texts_per_sec": total_texts / total_time,
                "avg_server_processing_time": statistics.mean(server_times),
                "avg_request_time": statistics.mean(request_times),
                "p95_request_time": statistics.quantiles(request_times, n=20)[18] if len(request_times) > 20 else max(request_times),
                "concurrent_requests": concurrency,
                "total_texts": total_texts
            }
            
            print(f"    Total time: {total_time:.3f}s")
            print(f"    Throughput: {results[concurrency]['throughput_texts_per_sec']:.1f} texts/sec")
            print(f"    Avg request time: {results[concurrency]['avg_request_time']:.3f}s")
    
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")
    
    def create_performance_plots(self, results: Dict[str, Any]):
        """Create performance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = Path("performance_plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Throughput vs Sequence Length
            if "sequence_length" in results:
                seq_data = results["sequence_length"]
                seq_lengths = sorted(seq_data.keys())
                throughputs = [seq_data[seq_len]["throughput_texts_per_sec"] for seq_len in seq_lengths]
                tokens_per_sec = [seq_data[seq_len]["tokens_per_sec"] for seq_len in seq_lengths]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(seq_lengths, throughputs, 'b-o', linewidth=2, markersize=8)
                ax1.set_xlabel('Sequence Length (tokens)')
                ax1.set_ylabel('Throughput (texts/sec)')
                ax1.set_title('Throughput vs Sequence Length')
                ax1.grid(True, alpha=0.3)
                ax1.set_xscale('log')
                
                ax2.plot(seq_lengths, tokens_per_sec, 'r-o', linewidth=2, markersize=8)
                ax2.set_xlabel('Sequence Length (tokens)')
                ax2.set_ylabel('Tokens processed per second')
                ax2.set_title('Token Processing Rate vs Sequence Length')
                ax2.grid(True, alpha=0.3)
                ax2.set_xscale('log')
                
                plt.tight_layout()
                plt.savefig(plots_dir / "sequence_length_performance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Throughput vs Batch Size
            if "batch_size" in results:
                batch_data = results["batch_size"]
                batch_sizes = sorted(batch_data.keys())
                throughputs = [batch_data[bs]["throughput_texts_per_sec"] for bs in batch_sizes]
                latencies = [batch_data[bs]["avg_request_latency"] for bs in batch_sizes]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(batch_sizes, throughputs, 'g-o', linewidth=2, markersize=8)
                ax1.set_xlabel('Batch Size')
                ax1.set_ylabel('Throughput (texts/sec)')
                ax1.set_title('Throughput vs Batch Size')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(batch_sizes, latencies, 'orange', marker='o', linewidth=2, markersize=8)
                ax2.set_xlabel('Batch Size')
                ax2.set_ylabel('Average Latency (seconds)')
                ax2.set_title('Latency vs Batch Size')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "batch_size_performance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 3: Concurrent Load Performance
            if "concurrent_load" in results:
                conc_data = results["concurrent_load"]
                concurrencies = sorted(conc_data.keys())
                throughputs = [conc_data[c]["throughput_texts_per_sec"] for c in concurrencies]
                avg_times = [conc_data[c]["avg_request_time"] for c in concurrencies]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(concurrencies, throughputs, 'purple', marker='o', linewidth=2, markersize=8)
                ax1.set_xlabel('Concurrent Requests')
                ax1.set_ylabel('Throughput (texts/sec)')
                ax1.set_title('Throughput vs Concurrent Load')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(concurrencies, avg_times, 'brown', marker='o', linewidth=2, markersize=8)
                ax2.set_xlabel('Concurrent Requests')
                ax2.set_ylabel('Average Request Time (seconds)')
                ax2.set_title('Request Time vs Concurrent Load')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "concurrent_load_performance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"📊 Performance plots saved to {plots_dir}/")
            
        except ImportError:
            print("⚠️ matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")


async def main():
    """Main performance test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive performance test for encoder server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, 
                       default=[128, 256, 512, 1024, 2048, 4096, 8192], 
                       help="Sequence lengths to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, 
                       default=[1, 2, 4, 8, 16, 32, 64], 
                       help="Batch sizes to test")
    parser.add_argument("--concurrent-requests", nargs="+", type=int,
                       default=[1, 2, 4, 8, 16, 32],
                       help="Concurrent request levels to test")
    parser.add_argument("--rounds", type=int, default=3, help="Number of test rounds")
    parser.add_argument("--samples-per-length", type=int, default=8, help="Samples per sequence length")
    parser.add_argument("--output", default="performance_results.json", help="Output file for results")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive performance testing...")
    print(f"Server URL: {args.url}")
    print(f"Sequence lengths: {args.sequence_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Concurrent requests: {args.concurrent_requests}")
    
    async with PerformanceTester(args.url) as tester:
        # Check server health
        try:
            async with tester.session.get(f"{args.url}/health") as response:
                if response.status != 200:
                    print("❌ Server is not healthy")
                    sys.exit(1)
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            sys.exit(1)
        
        print("✅ Server is healthy")
        
        all_results = {}
        
        # Generate test data
        print(f"\n📝 Generating test data...")
        test_data = tester.generate_test_data(args.sequence_lengths, args.samples_per_length)
        
        # Test 1: Sequence Length Performance
        print(f"\n" + "="*60)
        print("TEST 1: SEQUENCE LENGTH PERFORMANCE")
        print("="*60)
        
        seq_results = await tester.test_sequence_length_performance(
            test_data, batch_size=8, rounds=args.rounds
        )
        all_results["sequence_length"] = seq_results
        
        # Test 2: Batch Size Performance (using medium length sequences)
        print(f"\n" + "="*60)
        print("TEST 2: BATCH SIZE PERFORMANCE")
        print("="*60)
        
        # Use 512-token sequences for batch size testing
        medium_texts = test_data.get(512, test_data[sorted(test_data.keys())[len(test_data)//2]])
        batch_results = await tester.test_batch_size_performance(
            medium_texts * 8,  # Replicate to have enough data
            args.batch_sizes, 
            rounds=args.rounds
        )
        all_results["batch_size"] = batch_results
        
        # Test 3: Concurrent Load Performance
        print(f"\n" + "="*60)
        print("TEST 3: CONCURRENT LOAD PERFORMANCE")
        print("="*60)
        
        concurrent_results = await tester.test_concurrent_load(
            medium_texts,
            args.concurrent_requests,
            batch_size=4
        )
        all_results["concurrent_load"] = concurrent_results
        
        # Save results
        tester.save_results(all_results, args.output)
        
        # Create plots
        if not args.skip_plots:
            tester.create_performance_plots(all_results)
        
        # Summary
        print(f"\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if "sequence_length" in all_results:
            seq_data = all_results["sequence_length"]
            print(f"\n📏 Sequence Length Performance:")
            for seq_len in sorted(seq_data.keys()):
                data = seq_data[seq_len]
                print(f"  {seq_len:4d} tokens: {data['throughput_texts_per_sec']:6.1f} texts/sec, {data['tokens_per_sec']:8.0f} tokens/sec")
        
        if "batch_size" in all_results:
            batch_data = all_results["batch_size"]
            print(f"\n📦 Batch Size Performance:")
            for batch_size in sorted(batch_data.keys()):
                data = batch_data[batch_size]
                print(f"  Batch {batch_size:2d}: {data['throughput_texts_per_sec']:6.1f} texts/sec, {data['avg_request_latency']*1000:5.1f}ms latency")
        
        if "concurrent_load" in all_results:
            conc_data = all_results["concurrent_load"]
            print(f"\n🚀 Concurrent Load Performance:")
            for concurrency in sorted(conc_data.keys()):
                data = conc_data[concurrency]
                print(f"  {concurrency:2d} concurrent: {data['throughput_texts_per_sec']:6.1f} texts/sec, {data['avg_request_time']*1000:5.1f}ms avg time")
        
        print(f"\n✅ Performance testing completed!")
        print(f"📄 Detailed results saved to: {args.output}")
        if not args.skip_plots:
            print(f"📊 Performance plots saved to: performance_plots/")


if __name__ == "__main__":
    asyncio.run(main())