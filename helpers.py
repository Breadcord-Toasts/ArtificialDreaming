import datetime
import io
from logging import Logger
from typing import NamedTuple, Any

import aiohttp
import discord
from discord.ext import commands

from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI

__all__ = [
    "fetch_image",
    "APIPackage",
    "report_error",
    "resize_to_match_area",
    "LongLastingView",
    "map_flag_emojis",
    "format_embed_desc",
    "bool_emoji",
    "readable_delta",
]


async def fetch_image(image: str | bytes, session: aiohttp.ClientSession) -> io.BytesIO:
    if not image:
        raise ValueError("No image provided. Was empty data received?")
    if isinstance(image, bytes):
        return io.BytesIO(image)
    elif isinstance(image, str):
        async with session.get(image) as response:
            if not response.ok:
                raise RuntimeError(f"Failed to fetch image: {response.status} {response.reason} ({image})")
            return io.BytesIO(await response.read())
    raise ValueError(f"Invalid image type: {type(image)}")


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
        try:
            await context.response.send_message(embed=embed, ephemeral=True)
        except discord.InteractionResponded:
            await context.followup.send(embed=embed, ephemeral=True)
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


class LongLastingView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=60*60)


def map_flag_emojis(*info: tuple[Any, str | tuple[str | None, str | None]]) -> list[str]:
    mapped = []
    for check, emojis in info:
        result: str | None = (
            (emojis if check else None)
            if isinstance(emojis, str) else
            (emojis[0] if check else emojis[1])
        )
        if result is not None:
            mapped.append(result)
    return mapped


def format_embed_desc(items: dict[str, Any | None]) -> str:
    return "\n".join(
        f"**{key}:** {value}" if key else value
        for key, value in items.items()
        if value is not None
    )


def readable_delta(delta: datetime.timedelta) -> str:
    if delta.days >= 5:
        return f"{delta.days} days"
    elif delta.days >= 1:
        return f"{delta.days} days, {delta.seconds // 3600} hours"
    elif delta.seconds >= 3600:
        return f"{delta.seconds // 3600} hours, {delta.seconds % 3600 // 60} minutes"
    elif delta.seconds >= 60:
        return f"{delta.seconds // 60} minutes, {delta.seconds % 60} seconds"
    return f"{delta.seconds} seconds"


def bool_emoji(value: bool) -> str:
    return ":white_check_mark:" if value else ":x:"
