#!/usr/bin/env python3
"""
Device detection and optimization utilities for Weave scorers.
Supports CUDA, MPS (Apple Silicon), and CPU with automatic fallback.
"""

import logging
import platform
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class DeviceType(str, Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

class DeviceManager:
    """Manages device detection and optimization for Weave scorers."""
    
    def __init__(self):
        self.device_info = self._detect_devices()
        self.optimal_device = self._get_optimal_device()
    
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available devices and their capabilities."""
        device_info = {
            "available_devices": [],
            "cuda_available": False,
            "mps_available": False,
            "cpu_info": self._get_cpu_info()
        }
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                device_info["cuda_available"] = True
                device_info["available_devices"].append(DeviceType.CUDA)
                device_info["cuda_info"] = self._get_cuda_info()
                logger.info(f"CUDA detected: {device_info['cuda_info']['device_count']} device(s)")
        except ImportError:
            logger.warning("PyTorch not available, cannot detect CUDA")
        except Exception as e:
            logger.warning(f"Error detecting CUDA: {e}")
        
        # Check MPS availability (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info["mps_available"] = True
                device_info["available_devices"].append(DeviceType.MPS)
                device_info["mps_info"] = self._get_mps_info()
                logger.info("MPS (Apple Silicon GPU) detected")
        except Exception as e:
            logger.debug(f"MPS not available: {e}")
        
        # CPU is always available
        device_info["available_devices"].append(DeviceType.CPU)
        logger.info(f"CPU detected: {device_info['cpu_info']['cores']} cores")
        
        return device_info
    
    def _get_optimal_device(self) -> DeviceType:
        """Get the optimal device based on availability and performance."""
        # Priority: CUDA > MPS > CPU
        if self.device_info["cuda_available"]:
            return DeviceType.CUDA
        elif self.device_info["mps_available"]:
            return DeviceType.MPS
        else:
            return DeviceType.CPU
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        try:
            import torch
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                "memory_total": torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else 0,
                "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}" if torch.cuda.device_count() > 0 else "Unknown"
            }
            return cuda_info
        except Exception as e:
            logger.error(f"Error getting CUDA info: {e}")
            return {"error": str(e)}
    
    def _get_mps_info(self) -> Dict[str, Any]:
        """Get MPS device information."""
        try:
            import torch
            mps_info = {
                "available": torch.backends.mps.is_available(),
                "built": torch.backends.mps.is_built(),
                "platform": platform.machine(),
                "system": platform.system()
            }
            return mps_info
        except Exception as e:
            logger.error(f"Error getting MPS info: {e}")
            return {"error": str(e)}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            import psutil
            cpu_info = {
                "cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                "architecture": platform.machine(),
                "system": platform.system()
            }
        except ImportError:
            import multiprocessing
            cpu_info = {
                "cores": multiprocessing.cpu_count(),
                "logical_cores": multiprocessing.cpu_count(),
                "frequency": "Unknown",
                "architecture": platform.machine(),
                "system": platform.system()
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            cpu_info = {
                "cores": 1,
                "logical_cores": 1,
                "frequency": "Unknown",
                "architecture": platform.machine(),
                "system": platform.system(),
                "error": str(e)
            }
        
        return cpu_info
    
    def get_device(self, requested_device: Optional[str] = None) -> DeviceType:
        """Get the device to use, with optional override."""
        if requested_device and requested_device != "auto":
            # Validate requested device is available
            requested_type = DeviceType(requested_device.lower())
            if requested_type in self.device_info["available_devices"]:
                return requested_type
            else:
                logger.warning(f"Requested device {requested_device} not available, using {self.optimal_device}")
                return self.optimal_device
        
        return self.optimal_device
    
    def get_optimal_config(self, device: DeviceType) -> Dict[str, Any]:
        """Get optimal configuration for the specified device."""
        base_config = {
            "device": device.value,
            "enable_optimization": True
        }
        
        if device == DeviceType.CUDA:
            base_config.update({
                "max_batch_size": 64,
                "num_workers": 8,
                "batch_timeout": 0.05,  # Faster batching for GPU
                "enable_caching": True,
                "cache_size": 2000,
                "compile_model": True,  # Enable for PyTorch 2.0+
                "mixed_precision": True
            })
        elif device == DeviceType.MPS:
            base_config.update({
                "max_batch_size": 32,
                "num_workers": 6,
                "batch_timeout": 0.1,
                "enable_caching": True,
                "cache_size": 1500,
                "compile_model": False,  # MPS may not support all compile features
                "mixed_precision": False
            })
        else:  # CPU
            base_config.update({
                "max_batch_size": 16,
                "num_workers": self.device_info["cpu_info"]["cores"],
                "batch_timeout": 0.2,
                "enable_caching": True,
                "cache_size": 1000,
                "compile_model": False,
                "mixed_precision": False
            })
        
        return base_config
    
    def get_device_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the current device."""
        memory_info = {}
        
        if self.device_info["cuda_available"]:
            try:
                import torch
                memory_info["cuda"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "cached": torch.cuda.memory_reserved(),
                    "total": torch.cuda.get_device_properties(0).total_memory
                }
            except Exception as e:
                memory_info["cuda"] = {"error": str(e)}
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info["system"] = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except Exception as e:
            memory_info["system"] = {"error": str(e)}
        
        return memory_info
    
    def log_device_info(self):
        """Log comprehensive device information."""
        logger.info("=" * 50)
        logger.info("🖥️  Device Information")
        logger.info("=" * 50)
        logger.info(f"Optimal device: {self.optimal_device.value.upper()}")
        logger.info(f"Available devices: {[d.value for d in self.device_info['available_devices']]}")
        
        if self.device_info["cuda_available"]:
            cuda_info = self.device_info["cuda_info"]
            logger.info(f"🚀 CUDA: {cuda_info.get('device_name', 'Unknown')} ({cuda_info.get('memory_total', 0) // 1024**3}GB)")
        
        if self.device_info["mps_available"]:
            logger.info(f"🍎 MPS: Available on {self.device_info['mps_info'].get('platform', 'Unknown')}")
        
        cpu_info = self.device_info["cpu_info"]
        logger.info(f"💻 CPU: {cpu_info['cores']} cores ({cpu_info['logical_cores']} logical)")
        logger.info("=" * 50)

def get_device_manager() -> DeviceManager:
    """Get a global device manager instance."""
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager()
    return get_device_manager._instance