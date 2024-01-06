import io

import aiohttp
import discord
from discord.ext import commands, tasks

import breadcord
from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.civitai import ModelType
from .ai_horde.models.general import HordeRequestError
from .ai_horde.models.horde_meta import HordeNews
from .ai_horde.models.image import (
    Base64Image,
    GenericProcessedImageResult,
    ImageGenerationParams,
    ImageGenerationRequest,
    InterrogationRequest,
    InterrogationRequestForm,
    InterrogationType,
)


async def file_from_url(session: aiohttp.ClientSession, url: str) -> io.BytesIO:
    async with session.get(url) as response:
        return io.BytesIO(await response.read())


class ArtificialDreaming(breadcord.module.ModuleCog):
    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.generic_session: aiohttp.ClientSession = None  # type: ignore[assignment]
        self.horde: HordeAPI = None  # type: ignore[assignment]
        self.civitai: CivitAIAPI = None  # type: ignore[assignment]
        self.cache: Cache = None  # type: ignore[assignment]

    async def cog_load(self) -> None:
        common_headers = {
            "User-Agent": f"Breadcord {self.module.manifest.name}/{self.module.manifest.version}",
        }

        self.generic_session = aiohttp.ClientSession()
        self.horde = HordeAPI(
            aiohttp.ClientSession(headers=common_headers | {
                "apikey": self.settings.horde_api_key.value,
            }),
            logger=self.logger,
        )
        self.civitai = CivitAIAPI(
            aiohttp.ClientSession(headers=common_headers | {
                "Authorization": f"Bearer {self.settings.civitai_api_key.value}",
            }),
            logger=self.logger,
        )
        self.cache = Cache(
            session=self.generic_session,
            horde_api=self.horde,
            civitai_api=self.civitai,
            logger=self.logger,
            storage_path=self.module.storage_path / "cache",
            formatted_cache=self.bot.settings.debug.value,
        )

        self.update_cache.start()

    async def cog_unload(self) -> None:
        if self.generic_session is not None and not self.generic_session.closed:
            await self.generic_session.close()
        if self.horde.session is not None and not self.horde.session.closed:
            await self.horde.session.close()
        if self.civitai.session is not None and not self.civitai.session.closed:
            await self.civitai.session.close()

    @tasks.loop(hours=1)
    async def update_cache(self) -> None:
        await self.cache.update()

    @commands.hybrid_command()
    async def generate(self, ctx: commands.Context, prompt: str, negative_prompt: str | None = None) -> None:
        response = await ctx.reply("Generating image... Please wait.")

        try:
            finished_generation = await self.horde.generate_image(ImageGenerationRequest(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                params=ImageGenerationParams(
                ),
                r2=False,
            ))
        except HordeRequestError as horde_error:
            await response.edit(content=f"Error: {horde_error.message}")
            return

        await response.edit(
            content="Generated image.",
            attachments=[discord.File(
                finished_generation.generations[0].img.to_bytesio(),
                filename="image.webp",
            )],
        )

    @commands.hybrid_command()
    async def remove_bg(self, ctx: commands.Context, image_url: str) -> None:
        response = await ctx.reply("Interrogating... Please wait.")

        finished_interrogation = await self.horde.interrogate(InterrogationRequest(
            image_url=image_url,
            forms=[
                InterrogationRequestForm(name=InterrogationType.GFPGAN),
            ],
        ))
        result: GenericProcessedImageResult = finished_interrogation.forms[0].result

        await response.edit(
            content="Interrogated image.",
            attachments=[discord.File(
                await file_from_url(self.generic_session, result.image_url),
                filename="image.webp",
            )],
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
