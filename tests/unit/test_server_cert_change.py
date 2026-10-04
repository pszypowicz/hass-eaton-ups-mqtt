"""Tests for connecting to a broker whose certificate is or is not the pinned one."""

from __future__ import annotations

import contextlib
import socket
import ssl
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from custom_components.eaton_ups_mqtt.api import (
    EatonUpsClientAuthenticationError,
    EatonUpsMqttClient,
    EatonUpsMqttConfig,
    EatonUpsServerCertificateError,
)
from custom_components.eaton_ups_mqtt.certificates import generate_client_certificate

if TYPE_CHECKING:
    from collections.abc import Generator

MQTT_CONNACK_ACCEPTED = b"\x20\x02\x00\x00"


def _server_context(cert_pem: str, key_pem: str) -> ssl.SSLContext:
    """Build a TLS server context that presents the given certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with tempfile.TemporaryDirectory() as tmp:
        cert_path = Path(tmp) / "cert.pem"
        key_path = Path(tmp) / "key.pem"
        cert_path.write_text(cert_pem)
        key_path.write_text(key_pem)
        context.load_cert_chain(cert_path, key_path)
    return context


def _serve_one_client(listener: socket.socket, context: ssl.SSLContext) -> None:
    """Accept one client, answer its MQTT CONNECT, and wait for it to close."""
    conn, _ = listener.accept()
    with conn, contextlib.suppress(OSError):
        tls = context.wrap_socket(conn, server_side=True)
        tls.recv(1024)
        tls.sendall(MQTT_CONNACK_ACCEPTED)
        while tls.recv(1024):
            pass


class FakeBroker:
    """A one-shot TLS broker on localhost that accepts a single MQTT CONNECT."""

    def __init__(self, cert_pem: str, key_pem: str) -> None:
        """Start listening on a free localhost port."""
        self._context = _server_context(cert_pem, key_pem)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(
            target=_serve_one_client,
            args=(self._listener, self._context),
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the broker."""
        self._listener.close()
        self._thread.join(timeout=5)


@pytest.fixture
def broker_cert() -> tuple[str, str]:
    """Return the certificate and key the broker presents."""
    return generate_client_certificate("ups.example.local")


@pytest.fixture
def client_cert() -> tuple[str, str]:
    """Return the client certificate and key."""
    return generate_client_certificate("Home Assistant")


@pytest.fixture
def broker(broker_cert: tuple[str, str]) -> Generator[FakeBroker]:
    """Run a fake broker for the duration of a test."""
    fake = FakeBroker(*broker_cert)
    yield fake
    fake.close()


def _make_client(
    port: int, server_cert: str, client_cert: tuple[str, str]
) -> EatonUpsMqttClient:
    """Create a client that pins the given server certificate."""
    config = EatonUpsMqttConfig(
        host="127.0.0.1",
        port=port,
        server_cert=server_cert,
        client_cert=client_cert[0],
        client_key=client_cert[1],
    )
    return EatonUpsMqttClient(config, MagicMock())


@pytest.mark.usefixtures("socket_enabled")
class TestServerCertificateChange:
    """Tests for async_setup against a real TLS broker."""

    @pytest.mark.asyncio
    async def test_changed_server_cert_raises_server_cert_error(
        self, broker, client_cert
    ):
        """Test a broker certificate that differs from the pin raises at once."""
        stale_pin, _ = generate_client_certificate("ups.example.local")
        client = _make_client(broker.port, stale_pin, client_cert)

        try:
            with (
                patch(
                    "custom_components.eaton_ups_mqtt.api.MQTT_CONNECTION_ATTEMPTS",
                    1,
                ),
                pytest.raises(EatonUpsServerCertificateError) as exc_info,
            ):
                await client.async_setup()
        finally:
            await client.async_disconnect()

        assert isinstance(exc_info.value, EatonUpsClientAuthenticationError)
        assert "127.0.0.1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_changed_server_cert_does_not_start_network_loop(
        self, broker, client_cert
    ):
        """Test no paho network thread keeps retrying after the mismatch."""
        stale_pin, _ = generate_client_certificate("ups.example.local")
        client = _make_client(broker.port, stale_pin, client_cert)
        threads_before = set(threading.enumerate())

        try:
            with pytest.raises(EatonUpsServerCertificateError):
                await client.async_setup()
            new_threads = set(threading.enumerate()) - threads_before
            assert not [t for t in new_threads if t.name.startswith("paho-mqtt")]
        finally:
            await client.async_disconnect()

    @pytest.mark.asyncio
    async def test_pinned_server_cert_connects(self, broker, broker_cert, client_cert):
        """Test a broker certificate equal to the pin connects."""
        client = _make_client(broker.port, broker_cert[0], client_cert)

        try:
            await client.async_setup()
            assert client._mqtt_connected is True
        finally:
            await client.async_disconnect()
