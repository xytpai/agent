"""TLS configuration for the local web console.

Generated certificates are for local development, not public deployments.
Never fall back to plaintext HTTP if certificate loading fails.
"""

import os
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _generate_local_certificate(cert_path, key_path):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise RuntimeError(
            "Generating a local TLS certificate requires cryptography; "
            "install requirements.txt or supply --ssl-cert and --ssl-key."
        ) from exc
    import ipaddress

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address("::1")),
            ]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_bytes = cert.public_bytes(serialization.Encoding.PEM)

    # Never overwrite existing credentials. Keep generated private files owner-only.
    created = []
    try:
        for path, data in ((key_path, key_bytes), (cert_path, cert_bytes)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created.append(path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def create_ssl_context(cert_file=None, key_file=None, *, cert_dir):
    """Load a supplied PEM pair, or create/reuse a persistent localhost pair."""
    if bool(cert_file) != bool(key_file):
        raise ValueError("--ssl-cert and --ssl-key must be supplied together.")

    if cert_file:
        cert_path = Path(cert_file).expanduser()
        key_path = Path(key_file).expanduser()
    else:
        directory = Path(cert_dir)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        cert_path = directory / "localhost.crt"
        key_path = directory / "localhost.key"
        # lexists also detects dangling symlinks, which must not be overwritten.
        cert_exists = os.path.lexists(cert_path)
        key_exists = os.path.lexists(key_path)
        if cert_exists != key_exists:
            raise ValueError(
                f"Incomplete local TLS certificate pair in {directory}. "
                "Restore the missing file, or remove both files to regenerate."
            )
        if not cert_exists:
            _generate_local_certificate(cert_path, key_path)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context
