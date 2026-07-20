from __future__ import annotations

import logging

import uvicorn


def main() -> None:
    try:
        from dotenv import load_dotenv

        # Load the project .env (sibling to this file's project dir) explicitly so
        # the agent works regardless of the current working directory.
        project_env = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(project_env, override=True)
    except ImportError:
        pass
    from friday.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(
        "friday.web.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
