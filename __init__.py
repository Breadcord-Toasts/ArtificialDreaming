import io

import aiohttp
import discord
from discord.ext import commands, tasks

import breadcord
from .advanced_generate import APIPackage, GenerationSettingsView, get_settings_embed
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
    LoRA,
    Sampler,
    TextualInversion,
)


# === Big to do list ===
# TODO: Allow seeing previews for models, get data from https://github.com/Haidra-Org/AI-Horde-image-model-reference
#  This can also be used to provide descriptions to models in the generation command.
#  Make sure to cache it.


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

    @tasks.loop(minutes=10)
    async def update_cache(self) -> None:
        await self.cache.update()

    @commands.hybrid_command()
    async def generate(self, ctx: commands.Context, *, prompt: str, negative_prompt: str | None = None) -> None:
        response = await ctx.reply("Generating image... Please wait.")
        async for finished_image_pair in self.horde.generate_image(ImageGenerationRequest(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            models=["AlbedoBase XL (SDXL)"],
            params=ImageGenerationParams(
                width=1024,
                height=1024,
                sampler="k_dpmpp_sde",
                loras=[LoRA(identifier="247778", strength_model=1, is_version=True)],
                steps=8,
                cfg_scale=2,
                image_count=4,
            ),
            replacement_filter=True,
            r2=False,
        )):
            await response.edit(
                attachments=[
                    discord.File(
                        finished_generation.img.to_bytesio(),
                        filename="image.webp",
                    )
                    for finished_generation in finished_image_pair
                ],
            )
        await response.edit(content="Finished generation.")

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

    @commands.hybrid_command()
    async def advanced_generate(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> None:
        generation_request = ImageGenerationRequest(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            params=ImageGenerationParams(
                karras=True,
            ),
            replacement_filter=True,
        )
        apis = APIPackage(self.horde, self.civitai, self.cache, self.logger, self.generic_session)
        view = GenerationSettingsView(
            apis=apis,
            default_request=generation_request,
            author_id=ctx.author.id,
        )
        await ctx.reply(
            "Choose generation settings",
            view=view,
            embeds=await get_settings_embed(generation_request, apis),
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
