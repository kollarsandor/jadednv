# =======================================================================

#!/usr/bin/env python3
"""
NVIDIA AI/GPU Client - Production Implementation
Handles NVIDIA GPU monitoring, AI inference, and compute operations
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union
import httpx
import subprocess
from datetime import datetime
import threading

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

try:
    import nvidia_ml_py.nvml as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    logging.warning("nvidia-ml-py not available, some GPU features will be limited")

    # Create fallback nvml module for type checking
    class _FallbackNVML:
        @staticmethod
        def nvmlInit():
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetCount():
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetName(handle):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetUUID(handle):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetTemperature(handle, sensor):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlDeviceGetPowerUsage(handle):
            raise RuntimeError("NVML not available")
        @staticmethod
        def nvmlShutdown():
            pass
        NVML_TEMPERATURE_GPU = 0

    nvml = _FallbackNVML()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NVIDIAClient:
    """Production-ready NVIDIA client for GPU monitoring and AI operations"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")

        self.base_url = "https://api.nvcf.nvidia.com/v2/nvcf"
        self.nim_base_url = "https://integrate.api.nvidia.com/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Initialize NVML if available
        self.nvml_initialized = False
        if NVIDIA_ML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify NVIDIA API connection"""
        try:
            models = self.list_available_models()
            if models:
                logger.info("NVIDIA API connection verified successfully")
            else:
                logger.warning("NVIDIA API connection successful but no models found")
        except Exception as e:
            logger.error(f"NVIDIA API connection verification failed: {e}")
            raise

    # GPU Monitoring Functions
    def get_gpu_info(self) -> List[Dict]:
        """Get information about all GPUs"""
        gpus = []

        if self.nvml_initialized and NVIDIA_ML_AVAILABLE:
            try:
                device_count = nvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # Get basic info
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    uuid = nvml.nvmlDeviceGetUUID(handle).decode('utf-8')

                    # Get memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get temperature
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = -1

                    # Get power usage
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = -1

                    gpu_info = {
                        'index': i,
                        'name': name,
                        'uuid': uuid,
                        'memory': {
                            'total': mem_info.total,
                            'used': mem_info.used,
                            'free': mem_info.free,
                            'utilization_percent': (mem_info.used / mem_info.total) * 100
                        },
                        'utilization': {
                            'gpu_percent': util.gpu,
                            'memory_percent': util.memory
                        },
                        'temperature_c': temp,
                        'power_watts': power
                    }

                    gpus.append(gpu_info)

                logger.info(f"Retrieved info for {len(gpus)} GPUs")
                return gpus

            except Exception as e:
                logger.error(f"Failed to get GPU info via NVML: {e}")

        # Fallback to nvidia-smi if NVML not available
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 10:
                            gpu_info = {
                                'index': int(parts[0]) if parts[0] != '[Not Supported]' else -1,
                                'name': parts[1],
                                'uuid': parts[2],
                                'memory': {
                                    'total': int(parts[3]) * 1024 * 1024 if parts[3] != '[Not Supported]' else 0,
                                    'used': int(parts[4]) * 1024 * 1024 if parts[4] != '[Not Supported]' else 0,
                                    'free': int(parts[5]) * 1024 * 1024 if parts[5] != '[Not Supported]' else 0,
                                },
                                'utilization': {
                                    'gpu_percent': int(parts[6]) if parts[6] != '[Not Supported]' else -1,
                                    'memory_percent': int(parts[7]) if parts[7] != '[Not Supported]' else -1
                                },
                                'temperature_c': int(parts[8]) if parts[8] != '[Not Supported]' else -1,
                                'power_watts': float(parts[9]) if parts[9] != '[Not Supported]' else -1
                            }

                            if gpu_info['memory']['total'] > 0:
                                gpu_info['memory']['utilization_percent'] = (gpu_info['memory']['used'] / gpu_info['memory']['total']) * 100
                            else:
                                gpu_info['memory']['utilization_percent'] = 0

                            gpus.append(gpu_info)

                logger.info(f"Retrieved info for {len(gpus)} GPUs via nvidia-smi")
                return gpus
            else:
                logger.error(f"nvidia-smi failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to get GPU info via nvidia-smi: {e}")

        return gpus

    def monitor_gpu_usage(self, duration_seconds: int = 60, interval_seconds: int = 5) -> List[Dict]:
        """Monitor GPU usage over time"""
        samples = []
        start_time = time.time()

        try:
            while time.time() - start_time < duration_seconds:
                timestamp = datetime.now().isoformat()
                gpu_info = self.get_gpu_info()

                sample = {
                    'timestamp': timestamp,
                    'gpus': gpu_info
                }
                samples.append(sample)

                logger.info(f"GPU monitoring sample {len(samples)} collected")
                time.sleep(interval_seconds)

            logger.info(f"GPU monitoring completed: {len(samples)} samples collected")
            return samples

        except Exception as e:
            logger.error(f"GPU monitoring failed: {e}")
            raise

    # NVIDIA AI API Functions
    def list_available_models(self) -> List[Dict]:
        """List available AI models"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.nim_base_url}/models",
                    headers=self.headers
                )

                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get('data', [])
                    logger.info(f"Retrieved {len(models)} available models")
                    return models
                else:
                    logger.warning(f"Failed to get models: {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            raise

    def generate_text(self, model: str, prompt: str, max_tokens: int = 1024,
                     temperature: float = 0.7, top_p: float = 1.0) -> Dict:
        """Generate text using NVIDIA AI models"""
        try:
            payload = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'stream': False
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.nim_base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"Generated text using model {model}")
                return result

        except Exception as e:
            logger.error(f"Failed to generate text with model {model}: {e}")
            raise

    def generate_embeddings(self, model: str, texts: List[str]) -> Dict:
        """Generate embeddings for text"""
        try:
            payload = {
                'model': model,
                'input': texts
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.nim_base_url}/embeddings",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"Generated embeddings for {len(texts)} texts using model {model}")
                return result

        except Exception as e:
            logger.error(f"Failed to generate embeddings with model {model}: {e}")
            raise

    def generate_image(self, prompt: str, model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                      width: int = 1024, height: int = 1024, steps: int = 50) -> Dict:
        """Generate images using NVIDIA AI models"""
        try:
            payload = {
                'text_prompts': [
                    {
                        'text': prompt,
                        'weight': 1.0
                    }
                ],
                'width': width,
                'height': height,
                'steps': steps,
                'samples': 1,
                'cfg_scale': 7.5
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/exec/{model}",
                    headers=self.headers,
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"Generated image using model {model}")
                return result

        except Exception as e:
            logger.error(f"Failed to generate image with model {model}: {e}")
            raise

    def run_inference(self, model: str, inputs: Dict) -> Dict:
        """Run inference on NVIDIA models"""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/exec/{model}",
                    headers=self.headers,
                    json=inputs,
                    timeout=120.0
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"Ran inference on model {model}")
                return result

        except Exception as e:
            logger.error(f"Failed to run inference on model {model}: {e}")
            raise

    # GPU Memory Management
    def clear_gpu_memory(self, gpu_index: Optional[int] = None) -> bool:
        """Clear GPU memory"""
        try:
            if gpu_index is not None:
                # Clear specific GPU
                cmd = f"nvidia-smi --gpu-reset -i {gpu_index}"
            else:
                # Clear all GPUs
                cmd = "nvidia-smi --gpu-reset"

            result = subprocess.run(cmd.split(), capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"GPU memory cleared successfully")
                return True
            else:
                logger.error(f"Failed to clear GPU memory: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to clear GPU memory: {e}")
            return False

    def get_gpu_processes(self) -> List[Dict]:
        """Get processes running on GPUs"""
        processes = []

        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            process_info = {
                                'pid': int(parts[0]) if parts[0] != '[Not Supported]' else -1,
                                'process_name': parts[1],
                                'gpu_uuid': parts[2],
                                'used_memory_mb': int(parts[3]) if parts[3] != '[Not Supported]' else 0
                            }
                            processes.append(process_info)

                logger.info(f"Retrieved {len(processes)} GPU processes")
                return processes
            else:
                logger.error(f"Failed to get GPU processes: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to get GPU processes: {e}")

        return processes

    def set_gpu_performance_mode(self, gpu_index: int, mode: str = "max") -> bool:
        """Set GPU performance mode"""
        try:
            if mode == "max":
                persistence_mode = "1"
                power_limit = "max"
            elif mode == "normal":
                persistence_mode = "0"
                power_limit = "default"
            else:
                raise ValueError(f"Invalid performance mode: {mode}")

            # Set persistence mode
            cmd1 = f"nvidia-smi -i {gpu_index} -pm {persistence_mode}"
            result1 = subprocess.run(cmd1.split(), capture_output=True, text=True)

            if result1.returncode != 0:
                logger.error(f"Failed to set persistence mode: {result1.stderr}")
                return False

            logger.info(f"Set GPU {gpu_index} performance mode to {mode}")
            return True

        except Exception as e:
            logger.error(f"Failed to set GPU performance mode: {e}")
            return False

    def get_gpu_topology(self) -> Dict:
        """Get GPU topology and interconnect information"""
        try:
            result = subprocess.run([
                'nvidia-smi', 'topo', '-m'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                topology_info = {
                    'topology_matrix': result.stdout,
                    'timestamp': datetime.now().isoformat()
                }

                logger.info("Retrieved GPU topology information")
                return topology_info
            else:
                logger.error(f"Failed to get GPU topology: {result.stderr}")
                return {}

        except Exception as e:
            logger.error(f"Failed to get GPU topology: {e}")
            return {}

    def benchmark_gpu(self, gpu_index: int, duration_seconds: int = 60) -> Dict:
        """Run GPU benchmark"""
        try:
            # Start monitoring
            start_time = time.time()
            samples = []

            # Run a simple compute workload (matrix multiplication)
            benchmark_script = f"""
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_index}'

try:
    import cupy as cp
    # GPU benchmark with CuPy
    size = 4096
    a = cp.random.random((size, size), dtype=cp.float32)
    b = cp.random.random((size, size), dtype=cp.float32)

    start_time = time.time()
    for i in range(10):
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
    end_time = time.time()

    print(f"GPU benchmark completed: {{(end_time - start_time) / 10:.4f}} seconds per iteration")

except ImportError:
    # CPU fallback
    import numpy as np
    size = 2048
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)

    start_time = time.time()
    for i in range(5):
        c = np.dot(a, b)
    end_time = time.time()

    print(f"CPU benchmark completed: {{(end_time - start_time) / 5:.4f}} seconds per iteration")
"""

            # Save and run benchmark
            with open('/tmp/gpu_benchmark.py', 'w') as f:
                f.write(benchmark_script)

            result = subprocess.run([
                'python', '/tmp/gpu_benchmark.py'
            ], capture_output=True, text=True, timeout=duration_seconds + 30)

            # Clean up
            os.remove('/tmp/gpu_benchmark.py')

            benchmark_result = {
                'gpu_index': gpu_index,
                'duration_seconds': duration_seconds,
                'benchmark_output': result.stdout,
                'benchmark_error': result.stderr,
                'return_code': result.returncode,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"GPU {gpu_index} benchmark completed")
            return benchmark_result

        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            return {'error': str(e)}

    def close(self) -> None:
        """Clean up resources"""
        try:
            if self.nvml_initialized and NVIDIA_ML_AVAILABLE:
                nvml.nvmlShutdown()
            logger.info("NVIDIA client closed successfully")
        except Exception as e:
            logger.error(f"Failed to close NVIDIA client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
# =======================================================================


# =======================================================================
