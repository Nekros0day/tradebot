from __future__ import annotations

import secrets
import time
from cryptography.hazmat.primitives import serialization
import jwt


def format_jwt_uri(method: str, host: str, path: str) -> str:
    # Coinbase SDK style: "{METHOD} api.coinbase.com{path}" (NO query string)
    return f"{method.upper()} {host}{path}"


def build_rest_jwt(jwt_uri: str, key_name: str, private_key_pem: str) -> str:
    # Normalize common env formatting issues
    pem = private_key_pem.replace("\\n", "\n").replace("\r\n", "\n").strip() + "\n"

    private_key_bytes = pem.encode("utf-8")
    try:
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    except Exception as e:
        head = pem.splitlines()[0] if pem else ""
        tail = pem.splitlines()[-1] if pem else ""
        raise ValueError(
            f"Failed to load PEM private key. First line={head!r}, last line={tail!r}. "
            f"Make sure you created a Coinbase Advanced Trade (CDP) API key using ECDSA/ES256 "
            f"and that COINBASE_API_PRIVATE_KEY preserves newlines (use \\n escapes in .env). "
            f"Original error: {e}"
        ) from e

    payload = {
        "sub": key_name,
        "iss": "cdp",
        "nbf": int(time.time()) - 5,     # allow small clock skew
        "exp": int(time.time()) + 120,   # 2 minutes
        "uri": jwt_uri,
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": key_name, "nonce": secrets.token_hex()},
    )
    return token