# =======================================================================

#!/usr/bin/env python3
"""
Modal Cloud Computing Client - Production Implementation
Handles serverless function deployment and execution
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union, Callable
import modal
from modal import App, Function, Image, Secret, Volume
import asyncio
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalClient:
    """Production-ready Modal client for serverless computing"""

    def __init__(self, token_id: Optional[str] = None, token_secret: Optional[str] = None,
                 shell_token: Optional[str] = None):
        self.token_id = token_id or os.getenv('MODAL_TOKEN_ID')
        self.token_secret = token_secret or os.getenv('MODAL_TOKEN_SECRET')
        self.shell_token = shell_token or os.getenv('MODAL_SHELL_TOKEN')

        if not self.token_id or not self.token_secret:
            raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables are required")

        # Set environment variables for Modal SDK
        os.environ['MODAL_TOKEN_ID'] = self.token_id
        os.environ['MODAL_TOKEN_SECRET'] = self.token_secret

        self.app = App("modal-integration-client")
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Modal connection is working"""
        try:
            # Create a simple test function to verify connection
            @self.app.function()
            def test_connection():
                return "Modal connection successful"

            with self.app.run():
                result = test_connection.remote()
                if result == "Modal connection successful":
                    logger.info("Modal connection verified successfully")
                else:
                    raise ConnectionError("Failed to verify Modal connection")

        except Exception as e:
            logger.error(f"Modal connection verification failed: {e}")
            raise

    def create_function(self, func: Callable,
                       image: Optional[Image] = None,
                       secrets: Optional[List[Secret]] = None,
                       volumes: Optional[Dict[str, Volume]] = None,
                       mounts: Optional[List[Any]] = None,
                       gpu: Optional[str] = None,
                       memory: Optional[int] = None,
                       cpu: Optional[float] = None,
                       timeout: Optional[int] = None) -> Function:
        """Create and deploy a Modal function"""
        try:
            # Use default Python image if none provided
            if image is None:
                image = Image.debian_slim().pip_install(
                    "numpy", "requests", "pandas", "scikit-learn"
                )

            # Configure function parameters
            kwargs = {}
            if secrets:
                kwargs['secrets'] = secrets
            if volumes:
                kwargs['volumes'] = volumes
            if mounts:
                kwargs['mounts'] = mounts
            if gpu:
                kwargs['gpu'] = gpu
            if memory:
                kwargs['memory'] = memory
            if cpu:
                kwargs['cpu'] = cpu
            if timeout:
                kwargs['timeout'] = timeout

            # Create the function
            modal_func = self.app.function(image=image, **kwargs)(func)
            logger.info(f"Created Modal function: {func.__name__}")

            return modal_func

        except Exception as e:
            logger.error(f"Failed to create Modal function {func.__name__}: {e}")
            raise

    def deploy_app(self, app_name: Optional[str] = None) -> str:
        """Deploy the Modal app"""
        try:
            if app_name:
                self.app = self.app.with_name(app_name)

            # Deploy the app
            deployment = modal.deploy(self.app)
            logger.info(f"Deployed Modal app: {self.app.name}")
            return deployment

        except Exception as e:
            logger.error(f"Failed to deploy Modal app: {e}")
            raise

    def run_function(self, func: Function, *args, **kwargs) -> Any:
        """Run a Modal function"""
        try:
            with self.app.run():
                result = func.remote(*args, **kwargs)
                logger.info(f"Executed Modal function successfully")
                return result

        except Exception as e:
            logger.error(f"Failed to run Modal function: {e}")
            raise

    def run_function_async(self, func: Function, *args, **kwargs) -> Any:
        """Run a Modal function asynchronously"""
        try:
            with self.app.run():
                call = func.spawn(*args, **kwargs)
                logger.info(f"Spawned Modal function asynchronously")
                return call

        except Exception as e:
            logger.error(f"Failed to spawn Modal function: {e}")
            raise

    def create_secret(self, name: str, env_dict: Dict[str, str]) -> Secret:
        """Create a Modal secret"""
        try:
            secret = Secret.from_dict(env_dict)
            logger.info(f"Created Modal secret: {name}")
            return secret

        except Exception as e:
            logger.error(f"Failed to create Modal secret {name}: {e}")
            raise

    def create_volume(self, name: str) -> Volume:
        """Create a Modal volume"""
        try:
            volume = Volume.from_name(name, create_if_missing=True)
            logger.info(f"Created Modal volume: {name}")
            return volume

        except Exception as e:
            logger.error(f"Failed to create Modal volume {name}: {e}")
            raise

    def create_image(self, base_image: str = "python:3.11-slim") -> Image:
        """Create a custom Modal image"""
        try:
            image = Image.from_registry(base_image)
            logger.info(f"Created Modal image from: {base_image}")
            return image

        except Exception as e:
            logger.error(f"Failed to create Modal image: {e}")
            raise

    def install_packages(self, image: Image, packages: List[str]) -> Image:
        """Install packages in a Modal image"""
        try:
            updated_image = image.pip_install(*packages)
            logger.info(f"Installed packages in image: {packages}")
            return updated_image

        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            raise

    def install_apt_packages(self, image: Image, packages: List[str]) -> Image:
        """Install apt packages in a Modal image"""
        try:
            updated_image = image.apt_install(*packages)
            logger.info(f"Installed apt packages in image: {packages}")
            return updated_image

        except Exception as e:
            logger.error(f"Failed to install apt packages: {e}")
            raise

    def run_commands(self, image: Image, commands: List[str]) -> Image:
        """Run commands during image build"""
        try:
            updated_image = image.run_commands(*commands)
            logger.info(f"Executed commands in image: {len(commands)} commands")
            return updated_image

        except Exception as e:
            logger.error(f"Failed to run commands in image: {e}")
            raise

    def create_webhook(self, func: Function, method: str = "POST") -> str:
        """Create a webhook endpoint for a function"""
        try:
            # Add webhook decorator to function
            webhook_func = self.app.webhook(method=method)(func.raw_f)
            logger.info(f"Created webhook endpoint for function")
            return webhook_func

        except Exception as e:
            logger.error(f"Failed to create webhook: {e}")
            raise

    def create_schedule(self, func: Function, schedule: str) -> Function:
        """Create a scheduled function"""
        try:
            from modal import Cron

            # Parse schedule string and create cron
            if schedule.startswith("cron("):
                # Extract cron expression
                cron_expr = schedule[5:-1]
                scheduled_func = self.app.function(schedule=Cron(cron_expr))(func.raw_f)
            else:
                # Simple interval schedule
                if schedule.endswith('m'):
                    minutes = int(schedule[:-1])
                    scheduled_func = self.app.function(schedule=modal.Period(minutes=minutes))(func.raw_f)
                elif schedule.endswith('h'):
                    hours = int(schedule[:-1])
                    scheduled_func = self.app.function(schedule=modal.Period(hours=hours))(func.raw_f)
                else:
                    raise ValueError(f"Invalid schedule format: {schedule}")

            logger.info(f"Created scheduled function with schedule: {schedule}")
            return scheduled_func

        except Exception as e:
            logger.error(f"Failed to create scheduled function: {e}")
            raise

    def create_gpu_function(self, func: Callable, gpu_type: str = "T4",
                           gpu_count: int = 1, **kwargs) -> Function:
        """Create a GPU-enabled function"""
        try:
            # Configure GPU
            if gpu_type.upper() == "T4":
                gpu_config = modal.gpu.T4(count=gpu_count)
            elif gpu_type.upper() == "A10G":
                gpu_config = modal.gpu.A10G(count=gpu_count)
            elif gpu_type.upper() == "A100":
                gpu_config = modal.gpu.A100(count=gpu_count, size="40GB")
            elif gpu_type.upper() == "H100":
                gpu_config = modal.gpu.H100(count=gpu_count)
            else:
                raise ValueError(f"Unsupported GPU type: {gpu_type}")

            # Create GPU function
            gpu_func = self.app.function(gpu=gpu_config, **kwargs)(func)
            logger.info(f"Created GPU function with {gpu_type} x{gpu_count}")

            return gpu_func

        except Exception as e:
            logger.error(f"Failed to create GPU function: {e}")
            raise

    def shell_access(self, container_id: Optional[str] = None) -> None:
        """Access Modal shell"""
        try:
            if self.shell_token:
                cmd = ["modal", "shell", self.shell_token]
            elif container_id:
                cmd = ["modal", "shell", container_id]
            else:
                raise ValueError("Either shell_token or container_id is required")

            # Execute shell command
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Shell access initiated successfully")
                print(result.stdout)
            else:
                logger.error(f"Shell access failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to access shell: {e}")
            raise

    def list_apps(self) -> List[Dict]:
        """List deployed Modal apps"""
        try:
            # Use Modal CLI to list apps
            result = subprocess.run(
                ["modal", "app", "list"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Parse output (simplified - real implementation would parse properly)
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                apps = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            apps.append({
                                'name': parts[0],
                                'status': parts[1]
                            })

                logger.info(f"Listed {len(apps)} Modal apps")
                return apps
            else:
                logger.error(f"Failed to list apps: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"Failed to list Modal apps: {e}")
            raise

    def stop_app(self, app_name: str) -> bool:
        """Stop a deployed Modal app"""
        try:
            result = subprocess.run(
                ["modal", "app", "stop", app_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Stopped Modal app: {app_name}")
                return True
            else:
                logger.error(f"Failed to stop app {app_name}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to stop Modal app {app_name}: {e}")
            raise

    def get_logs(self, app_name: str, function_name: Optional[str] = None) -> str:
        """Get logs from Modal app/function"""
        try:
            cmd = ["modal", "app", "logs", app_name]
            if function_name:
                cmd.extend(["--function", function_name])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Retrieved logs for {app_name}")
                return result.stdout
            else:
                logger.error(f"Failed to get logs: {result.stderr}")
                return ""

        except Exception as e:
            logger.error(f"Failed to get Modal logs: {e}")
            raise

    def create_class_function(self, cls: type, **kwargs) -> modal.Cls:
        """Create a Modal class with methods"""
        try:
            modal_cls = self.app.cls(**kwargs)(cls)
            logger.info(f"Created Modal class: {cls.__name__}")
            return modal_cls

        except Exception as e:
            logger.error(f"Failed to create Modal class {cls.__name__}: {e}")
            raise

    def serve_locally(self, port: int = 8000) -> None:
        """Serve Modal app locally for development"""
        try:
            result = subprocess.run(
                ["modal", "serve", "--port", str(port)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Started local Modal server on port {port}")
                print(result.stdout)
            else:
                logger.error(f"Failed to start local server: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to serve locally: {e}")
            raise

    def close(self) -> None:
        """Clean up resources"""
        try:
            logger.info("Modal client closed successfully")
        except Exception as e:
            logger.error(f"Failed to close Modal client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
# =======================================================================


# =======================================================================
