#!/usr/bin/env python3
"""Load variables from the project `.env` file into the process environment."""

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parent.joinpath(".env")
    load_dotenv(dotenv_path=env_path)
