import asyncio
from agents import LiveVisionAgent

async def main():
    agent = LiveVisionAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
