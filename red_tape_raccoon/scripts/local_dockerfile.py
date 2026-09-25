"""Write a Dockerfile variant that trusts a TLS-intercepting egress proxy's CA bundle.

Only needed on sandboxed build hosts; the official image is unchanged.
Usage: python local_dockerfile.py docker/Dockerfile docker/Dockerfile.local
"""

import sys

ANCHOR = "FROM python:3.12-slim AS base\n"
INJECT = ANCHOR + (
    "COPY docker/ccr-ca-bundle.crt /etc/ssl/ccr-ca-bundle.crt\n"
    "ENV SSL_CERT_FILE=/etc/ssl/ccr-ca-bundle.crt REQUESTS_CA_BUNDLE=/etc/ssl/ccr-ca-bundle.crt \\\n"
    "    CURL_CA_BUNDLE=/etc/ssl/ccr-ca-bundle.crt NODE_EXTRA_CA_CERTS=/etc/ssl/ccr-ca-bundle.crt \\\n"
    "    PIP_CERT=/etc/ssl/ccr-ca-bundle.crt UV_NATIVE_TLS=1\n"
)


def main(src: str, dst: str) -> None:
    text = open(src).read()
    if ANCHOR not in text:
        raise SystemExit(f"anchor not found in {src}: {ANCHOR!r}")
    open(dst, "w").write(text.replace(ANCHOR, INJECT, 1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
