# =======================================================================

#!/usr/bin/env python3
"""
IONOS Cloud Client - Production Implementation
Handles cloud infrastructure management and operations
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union
import httpx
from datetime import datetime
import base64
import hashlib
import hmac

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IONOSClient:
    """Production-ready IONOS Cloud client"""

    def __init__(self, token_id: Optional[str] = None):
        self.token_id = token_id or os.getenv('IONOS_TOKEN_ID')
        if not self.token_id:
            raise ValueError("IONOS_TOKEN_ID environment variable is required")

        self.base_url = "https://api.ionos.com/cloudapi/v6"
        self.headers = {
            'Authorization': f'Bearer {self.token_id}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify IONOS API connection"""
        try:
            datacenters = self.list_datacenters()
            if isinstance(datacenters, list):
                logger.info("IONOS connection verified successfully")
            else:
                raise ConnectionError("Failed to verify IONOS connection")
        except Exception as e:
            logger.error(f"IONOS connection verification failed: {e}")
            raise

    # Datacenter Management
    def list_datacenters(self) -> List[Dict]:
        """List all datacenters"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                datacenters = data.get('items', [])
                logger.info(f"Retrieved {len(datacenters)} datacenters")
                return datacenters

        except Exception as e:
            logger.error(f"Failed to list datacenters: {e}")
            raise

    def create_datacenter(self, name: str, location: str, description: str = "") -> Dict:
        """Create a new datacenter"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'location': location,
                    'description': description
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                datacenter = response.json()

                logger.info(f"Created datacenter: {name} in {location}")
                return datacenter

        except Exception as e:
            logger.error(f"Failed to create datacenter {name}: {e}")
            raise

    def get_datacenter(self, datacenter_id: str) -> Dict:
        """Get datacenter details"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                datacenter = response.json()

                logger.info(f"Retrieved datacenter: {datacenter_id}")
                return datacenter

        except Exception as e:
            logger.error(f"Failed to get datacenter {datacenter_id}: {e}")
            raise

    def delete_datacenter(self, datacenter_id: str) -> bool:
        """Delete a datacenter"""
        try:
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/datacenters/{datacenter_id}",
                    headers=self.headers
                )
                response.raise_for_status()

                logger.info(f"Deleted datacenter: {datacenter_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete datacenter {datacenter_id}: {e}")
            raise

    # Server Management
    def list_servers(self, datacenter_id: str) -> List[Dict]:
        """List servers in a datacenter"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                servers = data.get('items', [])
                logger.info(f"Retrieved {len(servers)} servers in datacenter {datacenter_id}")
                return servers

        except Exception as e:
            logger.error(f"Failed to list servers in datacenter {datacenter_id}: {e}")
            raise

    def create_server(self, datacenter_id: str, name: str, cores: int, ram: int,
                     availability_zone: str = "AUTO", cpu_family: str = "AMD_OPTERON") -> Dict:
        """Create a new server"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'cores': cores,
                    'ram': ram * 1024,  # Convert GB to MB
                    'availabilityZone': availability_zone,
                    'cpuFamily': cpu_family,
                    'type': 'ENTERPRISE'
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                server = response.json()

                logger.info(f"Created server: {name} with {cores} cores and {ram}GB RAM")
                return server

        except Exception as e:
            logger.error(f"Failed to create server {name}: {e}")
            raise

    def get_server(self, datacenter_id: str, server_id: str) -> Dict:
        """Get server details"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                server = response.json()

                logger.info(f"Retrieved server: {server_id}")
                return server

        except Exception as e:
            logger.error(f"Failed to get server {server_id}: {e}")
            raise

    def start_server(self, datacenter_id: str, server_id: str) -> bool:
        """Start a server"""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/start",
                    headers=self.headers
                )
                response.raise_for_status()

                logger.info(f"Started server: {server_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to start server {server_id}: {e}")
            raise

    def stop_server(self, datacenter_id: str, server_id: str, force: bool = False) -> bool:
        """Stop a server"""
        try:
            endpoint = f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/stop"
            if force:
                endpoint = f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/reboot"

            with httpx.Client() as client:
                response = client.post(endpoint, headers=self.headers)
                response.raise_for_status()

                action = "force stopped" if force else "stopped"
                logger.info(f"Server {server_id} {action}")
                return True

        except Exception as e:
            logger.error(f"Failed to stop server {server_id}: {e}")
            raise

    def reboot_server(self, datacenter_id: str, server_id: str) -> bool:
        """Reboot a server"""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/reboot",
                    headers=self.headers
                )
                response.raise_for_status()

                logger.info(f"Rebooted server: {server_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to reboot server {server_id}: {e}")
            raise

    def delete_server(self, datacenter_id: str, server_id: str) -> bool:
        """Delete a server"""
        try:
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}",
                    headers=self.headers
                )
                response.raise_for_status()

                logger.info(f"Deleted server: {server_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete server {server_id}: {e}")
            raise

    # Volume Management
    def list_volumes(self, datacenter_id: str) -> List[Dict]:
        """List volumes in a datacenter"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}/volumes",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                volumes = data.get('items', [])
                logger.info(f"Retrieved {len(volumes)} volumes in datacenter {datacenter_id}")
                return volumes

        except Exception as e:
            logger.error(f"Failed to list volumes in datacenter {datacenter_id}: {e}")
            raise

    def create_volume(self, datacenter_id: str, name: str, size: int,
                     volume_type: str = "HDD", availability_zone: str = "AUTO") -> Dict:
        """Create a new volume"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'size': size,
                    'type': volume_type,
                    'availabilityZone': availability_zone
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/volumes",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                volume = response.json()

                logger.info(f"Created volume: {name} ({size}GB, {volume_type})")
                return volume

        except Exception as e:
            logger.error(f"Failed to create volume {name}: {e}")
            raise

    def attach_volume(self, datacenter_id: str, server_id: str, volume_id: str) -> Dict:
        """Attach volume to server"""
        try:
            payload = {
                'id': volume_id
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/volumes",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                attachment = response.json()

                logger.info(f"Attached volume {volume_id} to server {server_id}")
                return attachment

        except Exception as e:
            logger.error(f"Failed to attach volume {volume_id} to server {server_id}: {e}")
            raise

    def detach_volume(self, datacenter_id: str, server_id: str, volume_id: str) -> bool:
        """Detach volume from server"""
        try:
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/datacenters/{datacenter_id}/servers/{server_id}/volumes/{volume_id}",
                    headers=self.headers
                )
                response.raise_for_status()

                logger.info(f"Detached volume {volume_id} from server {server_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to detach volume {volume_id} from server {server_id}: {e}")
            raise

    # Network Management
    def list_lans(self, datacenter_id: str) -> List[Dict]:
        """List LANs in a datacenter"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}/lans",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                lans = data.get('items', [])
                logger.info(f"Retrieved {len(lans)} LANs in datacenter {datacenter_id}")
                return lans

        except Exception as e:
            logger.error(f"Failed to list LANs in datacenter {datacenter_id}: {e}")
            raise

    def create_lan(self, datacenter_id: str, name: str, public: bool = False) -> Dict:
        """Create a new LAN"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'public': public
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/lans",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                lan = response.json()

                lan_type = "public" if public else "private"
                logger.info(f"Created {lan_type} LAN: {name}")
                return lan

        except Exception as e:
            logger.error(f"Failed to create LAN {name}: {e}")
            raise

    # Load Balancer Management
    def list_load_balancers(self, datacenter_id: str) -> List[Dict]:
        """List load balancers in a datacenter"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/datacenters/{datacenter_id}/loadbalancers",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                load_balancers = data.get('items', [])
                logger.info(f"Retrieved {len(load_balancers)} load balancers in datacenter {datacenter_id}")
                return load_balancers

        except Exception as e:
            logger.error(f"Failed to list load balancers in datacenter {datacenter_id}: {e}")
            raise

    def create_load_balancer(self, datacenter_id: str, name: str, ip: Optional[str] = None,
                           dhcp: bool = True) -> Dict:
        """Create a new load balancer"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'dhcp': dhcp
                }
            }

            if ip:
                payload['properties']['ip'] = ip

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/loadbalancers",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                load_balancer = response.json()

                logger.info(f"Created load balancer: {name}")
                return load_balancer

        except Exception as e:
            logger.error(f"Failed to create load balancer {name}: {e}")
            raise

    # IP Block Management
    def list_ip_blocks(self) -> List[Dict]:
        """List IP blocks"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/ipblocks",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                ip_blocks = data.get('items', [])
                logger.info(f"Retrieved {len(ip_blocks)} IP blocks")
                return ip_blocks

        except Exception as e:
            logger.error(f"Failed to list IP blocks: {e}")
            raise

    def reserve_ip_block(self, location: str, size: int, name: str = "") -> Dict:
        """Reserve an IP block"""
        try:
            payload = {
                'properties': {
                    'location': location,
                    'size': size,
                    'name': name
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/ipblocks",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                ip_block = response.json()

                logger.info(f"Reserved IP block: {size} IPs in {location}")
                return ip_block

        except Exception as e:
            logger.error(f"Failed to reserve IP block: {e}")
            raise

    # Snapshot Management
    def list_snapshots(self) -> List[Dict]:
        """List snapshots"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/snapshots",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                snapshots = data.get('items', [])
                logger.info(f"Retrieved {len(snapshots)} snapshots")
                return snapshots

        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            raise

    def create_snapshot(self, datacenter_id: str, volume_id: str,
                       name: str, description: str = "") -> Dict:
        """Create a volume snapshot"""
        try:
            payload = {
                'properties': {
                    'name': name,
                    'description': description
                }
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/datacenters/{datacenter_id}/volumes/{volume_id}/create-snapshot",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                snapshot = response.json()

                logger.info(f"Created snapshot: {name} for volume {volume_id}")
                return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot {name}: {e}")
            raise

    # Kubernetes Management
    def list_k8s_clusters(self) -> List[Dict]:
        """List Kubernetes clusters"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://api.ionos.com/containerregistries/clusters",
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                clusters = data.get('items', [])
                logger.info(f"Retrieved {len(clusters)} Kubernetes clusters")
                return clusters

        except Exception as e:
            logger.error(f"Failed to list Kubernetes clusters: {e}")
            raise

    # Monitoring and Logging
    def get_resource_usage(self, datacenter_id: str, resource_type: str = "server",
                          resource_id: Optional[str] = None, period: str = "1h") -> Dict:
        """Get resource usage metrics"""
        try:
            endpoint = f"{self.base_url}/datacenters/{datacenter_id}"
            if resource_type == "server" and resource_id:
                endpoint += f"/servers/{resource_id}"
            elif resource_type == "volume" and resource_id:
                endpoint += f"/volumes/{resource_id}"

            params = {'depth': '5'}

            with httpx.Client() as client:
                response = client.get(endpoint, headers=self.headers, params=params)
                response.raise_for_status()
                usage_data = response.json()

                logger.info(f"Retrieved usage data for {resource_type}")
                return usage_data

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            raise

    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://api.ionos.com/cloudapi/v6/um/users",
                    headers=self.headers
                )
                response.raise_for_status()
                account_info = response.json()

                logger.info("Retrieved account information")
                return account_info

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    def wait_for_request(self, request_href: str, timeout: int = 300,
                        poll_interval: int = 5) -> Dict:
        """Wait for a request to complete"""
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                with httpx.Client() as client:
                    response = client.get(request_href, headers=self.headers)
                    response.raise_for_status()
                    request_data = response.json()

                    status = request_data.get('metadata', {}).get('status', 'UNKNOWN')

                    if status == 'DONE':
                        logger.info("Request completed successfully")
                        return request_data
                    elif status == 'FAILED':
                        logger.error(f"Request failed: {request_data.get('metadata', {}).get('message', 'Unknown error')}")
                        return request_data

                    logger.info(f"Request status: {status}, waiting...")
                    time.sleep(poll_interval)

            logger.warning(f"Request timed out after {timeout} seconds")
            return {'status': 'TIMEOUT'}

        except Exception as e:
            logger.error(f"Failed to wait for request: {e}")
            raise
# =======================================================================


# =======================================================================
