import io
from logging import Logger
from typing import NamedTuple

import aiohttp
import discord
from discord.ext import commands

from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.image import Base64Image

__all__ = [
    "fetch_image",
    "APIPackage",
    "report_error",
    "resize_to_match_area",
]


async def fetch_image(image: Base64Image | str, session: aiohttp.ClientSession) -> io.BytesIO:
    if isinstance(image, Base64Image):
        return image.to_bytesio()
    async with session.get(image) as response:
        return io.BytesIO(await response.read())


class APIPackage(NamedTuple):
    horde: HordeAPI
    civitai: CivitAIAPI
    cache: Cache
    logger: Logger
    session: aiohttp.ClientSession


async def report_error(
    context: discord.Interaction | commands.Context | discord.Message,
    error: Exception,
    *,
    title: str | None = None,
) -> None:
    embed = discord.Embed(
        title=title or "An error occurred",
        description=str(error),
        colour=discord.Colour.red(),
    )

    if isinstance(context, discord.Interaction):
        await context.response.send_message(embed=embed, ephemeral=True)
        return
    elif isinstance(context, (commands.Context, discord.Message)):
        await context.reply(embed=embed)
        return

    raise ValueError(f"Invalid context type: {type(context)}")


def resize_to_match_area(aspect_ratio: tuple[int, int], target_area: int, multiple_of: int = 64) -> tuple[int, int]:
    current_area = aspect_ratio[0] * aspect_ratio[1]
    scale_factor = (target_area / current_area) ** 0.5
    new_width = int(aspect_ratio[0] * scale_factor)
    new_height = int(aspect_ratio[1] * scale_factor)
    new_width = ((new_width + multiple_of // 2) // multiple_of) * multiple_of
    new_height = ((new_height + multiple_of // 2) // multiple_of) * multiple_of
    return new_width, new_height
