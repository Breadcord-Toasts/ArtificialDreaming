import random

import aiohttp
import discord
from discord.ext import commands, tasks

import breadcord
from .advanced_generate import APIPackage, GenerationSettingsView, get_settings_embeds
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
    TextualInversion, CaptionResult,
)
from .helpers import fetch_image


# === Big to do list ===
# TODO: Allow seeing previews for models, get data from https://github.com/Haidra-Org/AI-Horde-image-model-reference
#  This can also be used to provide descriptions to models in the generation command.
#  Make sure to cache it.


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
        # if self.bot.settings.debug.value:
        #     # Run tests

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
    async def generate(
            self,
            ctx: commands.Context,
            *,
            prompt: str,
            negative_prompt: str | None = None,
            random_style: bool = False,
    ) -> None:
        style = random.choice(tuple(self.cache.styles.values()))

        response = await ctx.reply(
            "Generating image... Please wait. \n"
            + (f"Style: {style.name}" if random_style else "")
        )

        if random_style:
            request = ImageGenerationRequest(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                nsfw=True,
                params=ImageGenerationParams(image_count=4),
                replacement_filter=True,
            ).apply_style(style)
        else:
            request = ImageGenerationRequest(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                nsfw=True,
                models=["AlbedoBase XL (SDXL)", "Fustercluck"],
                params=ImageGenerationParams(
                    width=1024,
                    height=1024,
                    sampler=Sampler.K_DPMPP_SDE,
                    loras=[LoRA(identifier="246747", strength_model=1, is_version=True)],
                    steps=8,
                    cfg_scale=2.0,
                    image_count=4,
                ),
                replacement_filter=True,
            )

        async for finished_image_pair in self.horde.generate_image(request):
            await response.edit(
                attachments=[
                    discord.File(
                        fp=await fetch_image(finished_generation.img, self.generic_session),
                        filename="image.webp",
                    )
                    for finished_generation in finished_image_pair
                ],
            )
        await response.edit(
            content=(
                "Finished generation. \n"
                + (f"Style: {style.name}" if random_style else "")
            )
        )

    @commands.hybrid_command()
    async def describe(self, ctx: commands.Context, image_url: str) -> None:
        response = await ctx.reply("Requesting interrogation... Please wait.")

        finished_interrogation = await self.horde.interrogate(InterrogationRequest(
            image_url=image_url,
            forms=[InterrogationRequestForm(name=InterrogationType.CAPTION)],
        ))
        result: CaptionResult = finished_interrogation.forms[0].result

        await response.edit(content=result.caption)

    @commands.hybrid_command()
    async def advanced_generate(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> None:
        source_image = None
        if ctx.message.attachments:
            async with self.generic_session.get(ctx.message.attachments[0].url) as response:
                source_image = Base64Image(await response.read())

        generation_request = ImageGenerationRequest(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            source_image=source_image,
            params=ImageGenerationParams(
                karras=True,
            ),
            replacement_filter=True,
        )
        apis = APIPackage(self.horde, self.civitai, self.cache, self.logger, self.generic_session)
        view = GenerationSettingsView(apis=apis, default_request=generation_request, author_id=ctx.author.id)
        await ctx.reply(
            "Choose generation settings",
            view=view,
            embeds=await get_settings_embeds(generation_request, apis),
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
