import io

import aiohttp

from .ai_horde.models.image import Base64Image


async def fetch_image(image: Base64Image | str, session: aiohttp.ClientSession) -> io.BytesIO:
    if isinstance(image, Base64Image):
        return image.to_bytesio()
    async with session.get(image) as response:
        return io.BytesIO(await response.read())
