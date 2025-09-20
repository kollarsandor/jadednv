# =======================================================================

"""
Integration Services Module - Production Implementation
Provides unified access to all external service integrations
"""

from .dragonfly_client import DragonflyClient
from .upstash_client import UpstashRedisClient, UpstashSearchClient
from .modal_client import ModalClient
from .ionq_client import IonQClient
from .nvidia_client import NVIDIAClient
from .ionos_client import IONOSClient

__all__ = [
    'DragonflyClient',
    'UpstashRedisClient',
    'UpstashSearchClient',
    'ModalClient',
    'IonQClient',
    'NVIDIAClient',
    'IONOSClient',
    'UnifiedClient'
]

class UnifiedClient:
    """Unified client for all integrated services"""

    def __init__(self):
        """Initialize all service clients"""
        self._dragonfly = None
        self._upstash_redis = None
        self._upstash_search = None
        self._modal = None
        self._ionq = None
        self._nvidia = None
        self._ionos = None

    @property
    def dragonfly(self) -> DragonflyClient:
        """Get DragonflyDB client"""
        if self._dragonfly is None:
            self._dragonfly = DragonflyClient()
        return self._dragonfly

    @property
    def upstash_redis(self) -> UpstashRedisClient:
        """Get Upstash Redis client"""
        if self._upstash_redis is None:
            self._upstash_redis = UpstashRedisClient()
        return self._upstash_redis

    @property
    def upstash_search(self) -> UpstashSearchClient:
        """Get Upstash Search client"""
        if self._upstash_search is None:
            self._upstash_search = UpstashSearchClient()
        return self._upstash_search

    @property
    def modal(self) -> ModalClient:
        """Get Modal client"""
        if self._modal is None:
            self._modal = ModalClient()
        return self._modal

    @property
    def ionq(self) -> IonQClient:
        """Get IonQ client"""
        if self._ionq is None:
            self._ionq = IonQClient()
        return self._ionq

    @property
    def nvidia(self) -> NVIDIAClient:
        """Get NVIDIA client"""
        if self._nvidia is None:
            self._nvidia = NVIDIAClient()
        return self._nvidia

    @property
    def ionos(self) -> IONOSClient:
        """Get IONOS client"""
        if self._ionos is None:
            self._ionos = IONOSClient()
        return self._ionos

    def close_all(self):
        """Close all client connections"""
        clients = [
            self._dragonfly, self._upstash_redis, self._upstash_search,
            self._modal, self._ionq, self._nvidia, self._ionos
        ]

        for client in clients:
            if client is not None and hasattr(client, 'close'):
                try:
                    client.close()
                except Exception as e:
                    print(f"Error closing client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
# =======================================================================


# =======================================================================
