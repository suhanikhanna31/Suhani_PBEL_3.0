"""
Small shared utilities that don't belong to any one pipeline stage.
"""
import ssl


def allow_unverified_ssl_for_nltk_downloads() -> None:
    """
    NLTK's corpus downloader (used once, on first run, to fetch POS-tagging
    data for src/features/stylometry.py) fails on some machines/corporate
    networks with a certificate-verification error, because the default
    Python SSL context is stricter than the certificate bundle NLTK's
    download server presents.

    This relaxes *only* the default HTTPS context used for that one-time
    download, at process start. It intentionally does not touch requests
    made by `requests`/`httpx` elsewhere in the app (e.g. the watsonx.ai
    client), which keep normal certificate verification.
    """
    try:
        _create_unverified_context = ssl._create_unverified_context
    except AttributeError:
        return
    ssl._create_default_https_context = _create_unverified_context
