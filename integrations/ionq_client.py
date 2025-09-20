# =======================================================================

#!/usr/bin/env python3
"""
IonQ Quantum Computing Client - Production Implementation
Handles quantum circuit execution on IonQ quantum computers
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union
import httpx
from datetime import datetime, timezone
import uuid
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IonQClient:
    """Production-ready IonQ quantum computing client"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('IONQ_API_KEY')
        if not self.api_key:
            raise ValueError("IONQ_API_KEY environment variable is required")

        self.base_url = "https://api.ionq.co/v0.3"
        self.headers = {
            'Authorization': f'apikey {self.api_key}',
            'Content-Type': 'application/json'
        }

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify IonQ API connection"""
        try:
            backends = self.get_backends()
            if backends:
                logger.info("IonQ connection verified successfully")
            else:
                raise ConnectionError("Failed to retrieve IonQ backends")
        except Exception as e:
            logger.error(f"IonQ connection verification failed: {e}")
            raise

    def get_backends(self) -> List[Dict]:
        """Get available quantum backends"""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/backends", headers=self.headers)
                response.raise_for_status()
                backends = response.json()

                logger.info(f"Retrieved {len(backends)} IonQ backends")
                return backends

        except Exception as e:
            logger.error(f"Failed to get backends: {e}")
            raise

    def get_backend_info(self, backend_name: str) -> Dict:
        """Get detailed backend information"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/backends/{backend_name}",
                    headers=self.headers
                )
                response.raise_for_status()
                backend_info = response.json()

                logger.info(f"Retrieved backend info for {backend_name}")
                return backend_info

        except Exception as e:
            logger.error(f"Failed to get backend info for {backend_name}: {e}")
            raise

    def submit_job(self, circuit: Dict, backend: str = "simulator",
                  shots: int = 1000, name: Optional[str] = None) -> Dict:
        """Submit a quantum job"""
        try:
            job_data = {
                'target': backend,
                'shots': shots,
                'input': circuit
            }

            if name:
                job_data['name'] = name
            else:
                job_data['name'] = f"job-{uuid.uuid4().hex[:8]}"

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/jobs",
                    headers=self.headers,
                    json=job_data,
                    timeout=30.0
                )
                response.raise_for_status()
                job = response.json()

                logger.info(f"Submitted job {job['id']} to {backend}")
                return job

        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise

    def get_job(self, job_id: str) -> Dict:
        """Get job status and results"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/jobs/{job_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                job = response.json()

                logger.info(f"Retrieved job {job_id} status: {job.get('status', 'unknown')}")
                return job

        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise

    def list_jobs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict]:
        """List jobs with optional status filter"""
        try:
            params = {'limit': limit}
            if status:
                params['status'] = status

            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/jobs",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                jobs = response.json()

                logger.info(f"Retrieved {len(jobs)} jobs")
                return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a running job"""
        try:
            with httpx.Client() as client:
                response = client.put(
                    f"{self.base_url}/jobs/{job_id}/status/cancel",
                    headers=self.headers
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"Cancelled job {job_id}")
                return result

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise

    def wait_for_job(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict:
        """Wait for job completion with timeout"""
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                job = self.get_job(job_id)
                status = job.get('status', 'unknown')

                if status == 'completed':
                    logger.info(f"Job {job_id} completed successfully")
                    return job
                elif status == 'failed':
                    logger.error(f"Job {job_id} failed: {job.get('failure', {}).get('error', 'Unknown error')}")
                    return job
                elif status == 'cancelled':
                    logger.info(f"Job {job_id} was cancelled")
                    return job

                logger.info(f"Job {job_id} status: {status}, waiting...")
                time.sleep(poll_interval)

            logger.warning(f"Job {job_id} timed out after {timeout} seconds")
            return self.get_job(job_id)

        except Exception as e:
            logger.error(f"Failed to wait for job {job_id}: {e}")
            raise

    def create_circuit(self, num_qubits: int) -> Dict:
        """Create a basic quantum circuit structure"""
        try:
            circuit = {
                'qubits': num_qubits,
                'circuit': []
            }

            logger.info(f"Created quantum circuit with {num_qubits} qubits")
            return circuit

        except Exception as e:
            logger.error(f"Failed to create circuit: {e}")
            raise

    def add_gate(self, circuit: Dict, gate: str, target: Union[int, List[int]],
                control: Optional[int] = None, rotation: Optional[float] = None) -> Dict:
        """Add a gate to the quantum circuit"""
        try:
            gate_op = {'gate': gate, 'target': target}

            if control is not None:
                gate_op['control'] = control

            if rotation is not None:
                gate_op['rotation'] = rotation

            circuit['circuit'].append(gate_op)

            logger.debug(f"Added {gate} gate to circuit")
            return circuit

        except Exception as e:
            logger.error(f"Failed to add gate {gate}: {e}")
            raise

    def add_measurement(self, circuit: Dict, qubits: Optional[List[int]] = None) -> Dict:
        """Add measurements to the circuit"""
        try:
            if qubits is None:
                # Measure all qubits
                qubits = list(range(circuit['qubits']))

            for qubit in qubits:
                circuit['circuit'].append({
                    'gate': 'mz',
                    'target': qubit
                })

            logger.debug(f"Added measurements for qubits {qubits}")
            return circuit

        except Exception as e:
            logger.error(f"Failed to add measurements: {e}")
            raise

    def create_bell_state_circuit(self) -> Dict:
        """Create a Bell state preparation circuit"""
        try:
            circuit = self.create_circuit(2)
            circuit = self.add_gate(circuit, 'h', 0)  # Hadamard on qubit 0
            circuit = self.add_gate(circuit, 'cnot', 1, control=0)  # CNOT with control=0, target=1
            circuit = self.add_measurement(circuit)

            logger.info("Created Bell state circuit")
            return circuit

        except Exception as e:
            logger.error(f"Failed to create Bell state circuit: {e}")
            raise

    def create_ghz_state_circuit(self, num_qubits: int = 3) -> Dict:
        """Create a GHZ state preparation circuit"""
        try:
            circuit = self.create_circuit(num_qubits)

            # Hadamard on first qubit
            circuit = self.add_gate(circuit, 'h', 0)

            # CNOT gates to create entanglement
            for i in range(1, num_qubits):
                circuit = self.add_gate(circuit, 'cnot', i, control=0)

            circuit = self.add_measurement(circuit)

            logger.info(f"Created GHZ state circuit with {num_qubits} qubits")
            return circuit

        except Exception as e:
            logger.error(f"Failed to create GHZ state circuit: {e}")
            raise

    def create_quantum_fourier_transform(self, num_qubits: int) -> Dict:
        """Create a Quantum Fourier Transform circuit"""
        try:
            circuit = self.create_circuit(num_qubits)

            # QFT implementation
            for j in range(num_qubits):
                # Hadamard gate
                circuit = self.add_gate(circuit, 'h', j)

                # Controlled phase gates
                for k in range(j + 1, num_qubits):
                    rotation = 2 * 3.14159 / (2 ** (k - j + 1))
                    circuit = self.add_gate(circuit, 'rz', k, control=j, rotation=rotation)

            # Swap qubits to reverse order
            for i in range(num_qubits // 2):
                j = num_qubits - 1 - i
                circuit = self.add_gate(circuit, 'swap', [i, j])

            circuit = self.add_measurement(circuit)

            logger.info(f"Created QFT circuit with {num_qubits} qubits")
            return circuit

        except Exception as e:
            logger.error(f"Failed to create QFT circuit: {e}")
            raise

    def run_circuit(self, circuit: Dict, backend: str = "simulator",
                   shots: int = 1000, wait: bool = True) -> Dict:
        """Run a quantum circuit and optionally wait for results"""
        try:
            # Submit the job
            job = self.submit_job(circuit, backend=backend, shots=shots)
            job_id = job['id']

            if wait:
                # Wait for completion
                completed_job = self.wait_for_job(job_id)
                return completed_job
            else:
                return job

        except Exception as e:
            logger.error(f"Failed to run circuit: {e}")
            raise

    def get_calibration_data(self, backend_name: str) -> Dict:
        """Get calibration data for a backend"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/backends/{backend_name}/calibration",
                    headers=self.headers
                )
                response.raise_for_status()
                calibration = response.json()

                logger.info(f"Retrieved calibration data for {backend_name}")
                return calibration

        except Exception as e:
            logger.error(f"Failed to get calibration data for {backend_name}: {e}")
            raise

    def estimate_cost(self, circuit: Dict, backend: str, shots: int) -> Dict:
        """Estimate the cost of running a circuit"""
        try:
            job_data = {
                'target': backend,
                'shots': shots,
                'input': circuit
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/jobs/estimate-cost",
                    headers=self.headers,
                    json=job_data
                )
                response.raise_for_status()
                cost_estimate = response.json()

                logger.info(f"Estimated cost for {backend}: ${cost_estimate.get('cost', 'N/A')}")
                return cost_estimate

        except Exception as e:
            logger.error(f"Failed to estimate cost: {e}")
            raise

    def get_account_info(self) -> Dict:
        """Get account information and credits"""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/user", headers=self.headers)
                response.raise_for_status()
                account_info = response.json()

                logger.info("Retrieved account information")
                return account_info

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    def validate_circuit(self, circuit: Dict) -> Dict:
        """Validate a quantum circuit"""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/circuits/validate",
                    headers=self.headers,
                    json={'input': circuit}
                )
                response.raise_for_status()
                validation_result = response.json()

                logger.info("Circuit validation completed")
                return validation_result

        except Exception as e:
            logger.error(f"Failed to validate circuit: {e}")
            raise
# =======================================================================


# =======================================================================
