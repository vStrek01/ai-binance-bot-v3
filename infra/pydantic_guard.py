from pydantic import __version__ as pydantic_version


def ensure_pydantic_v2() -> None:
    """
    Ensure that Pydantic v2 is installed. Raise ImportError if not.

    This protects the project from accidentally running with a global
    Python installation that only has pydantic v1.
    """

    try:
        major = int(pydantic_version.split(".")[0])
    except Exception:
        # If parsing fails, let later usage/tests surface the incompatibility.
        return

    if major < 2:
        raise ImportError(
            f"Pydantic v2 is required, but v{pydantic_version} is installed. "
            "Activate the project virtualenv (.venv) and install the correct dependencies."
        )
