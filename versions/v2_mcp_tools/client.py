import asyncio
from pathlib import Path

from fastmcp import Client

client = Client(str(Path(__file__).parent / "server.py"))


async def main():
    async with client:
        tools = await client.list_tools()

        print("Available tools:")

        for tool in tools:
            print(tool.name)
            print(tool.description)


if __name__ == "__main__":
    asyncio.run(main())