"""Entry point for the glm-acp command."""

from .agent import run

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
