import asyncio
import time
import httpx
import uuid

async def fetch(client, msg):
    start = time.time()
    async with client.stream("GET", f"http://localhost:8000/chat/1daa30c3-e629-47e7-978c-93f1be446450/stream?message={msg}", timeout=120.0) as r:
        async for chunk in r.aiter_text():
            pass
    return time.time() - start

async def main():
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        results = await asyncio.gather(
            fetch(client, "test1"),
            fetch(client, "test2")
        )
        total = time.time() - start_time
        print(f"Req 1: {results[0]:.2f}s, Req 2: {results[1]:.2f}s, Total: {total:.2f}s")

asyncio.run(main())
