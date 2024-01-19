import random
import sqlite3

import aiohttp
import discord
from discord.ext import commands, tasks
# noinspection PyProtectedMember
from discord.user import _UserTag as UserTag

import breadcord
from .advanced_generate import GenerationSettingsView, get_settings_embeds
from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.civitai import ModelType
from .ai_horde.models.general import HordeRequestError
from .ai_horde.models.horde_meta import HordeNews
from .ai_horde.models.image import (
    Base64Image,
    CaptionResult,
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
from .helpers import APIPackage, fetch_image
from .login import LoginButtonView


# === Big to do list ===
# TODO: Allow seeing previews for models, get data from https://github.com/Haidra-Org/AI-Horde-image-model-reference
#  This can also be used to provide descriptions to models in the generation command.
#  Make sure to cache it.


class ArtificialDreaming(breadcord.module.ModuleCog):
    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        # All set in cog_load
        self.generic_session: aiohttp.ClientSession = None  # type: ignore[assignment]
        self.anon_horde: HordeAPI = None  # type: ignore[assignment]
        self.civitai: CivitAIAPI = None  # type: ignore[assignment]
        self.cache: Cache = None  # type: ignore[assignment]
        self.database: sqlite3.Connection = None  # type: ignore[assignment]
        self.db_cursor: sqlite3.Cursor = None  # type: ignore[assignment]

        self._common_headers = {
            "User-Agent": f"Breadcord {self.module.manifest.name}/{self.module.manifest.version}",
        }
        self.horde_apis: dict[int, HordeAPI] = {}

    async def cog_load(self) -> None:
        self.generic_session = aiohttp.ClientSession()
        self.anon_horde = HordeAPI(
            aiohttp.ClientSession(
                headers=self._common_headers | {
                    "apikey": "0000000000",
                },
            ),
            logger=self.logger,
        )
        self.civitai = CivitAIAPI(
            aiohttp.ClientSession(headers=self._common_headers | {
                "Authorization": f"Bearer {self.settings.civitai_api_key.value}",
            }),
            logger=self.logger,
        )
        self.cache = Cache(
            session=self.generic_session,
            horde_api=self.anon_horde,
            civitai_api=self.civitai,
            logger=self.logger,
            storage_path=self.module.storage_path / "cache",
            formatted_cache=self.bot.settings.debug.value,
        )

        self.database = sqlite3.connect(self.module.storage_path / "database.db")
        self.db_cursor = self.database.cursor()
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                discord_id INTEGER PRIMARY KEY,
                horde_api_key TEXT,
                civitai_api_key TEXT
            )
            """,
        )
        self.database.commit()
        for discord_id, horde_api_key in self.db_cursor.execute(
            "SELECT discord_id, horde_api_key FROM users",
        ).fetchall():
            self.logger.debug(f"Logging in user {discord_id}")
            self.horde_apis[discord_id] = HordeAPI(
                aiohttp.ClientSession(headers=self._common_headers | {"apikey": horde_api_key}),
                logger=self.logger,
            )

        self.update_cache.start()
        # if self.bot.settings.debug.value:
        #     # Run tests

    async def cog_unload(self) -> None:
        self.update_cache.cancel()
        self.database.close()

        if self.generic_session is not None and not self.generic_session.closed:
            await self.generic_session.close()
        if self.civitai.session is not None and not self.civitai.session.closed:
            await self.civitai.session.close()
        if self.anon_horde is not None and not self.anon_horde.session.closed:
            await self.anon_horde.session.close()
        for api in self.horde_apis.values():
            if not api.session.closed:
                await api.session.close()

    @tasks.loop(minutes=10)
    async def update_cache(self) -> None:
        await self.cache.update()

    def horde_for(self, user: discord.User | discord.Member | int | UserTag) -> HordeAPI:
        if isinstance(user, UserTag):
            user = user.id
        api = self.horde_apis.get(user)
        if api is None:
            self.logger.debug("Using horde anonymously")
            return self.anon_horde
        return api

    @commands.hybrid_command()
    async def login(self, ctx: commands.Context) -> None:
        # TODO: If I ever implement CivitAI login, add it here.
        view = LoginButtonView()
        response = await ctx.reply(
            (
                "Make sure you have an account and API key for the [AI Horde](https://aihorde.net/). "
                "If you do not have an account, you can make one [here](https://aihorde.net/register). "
            ),
            view=view,
            ephemeral=True,
        )
        await view.wait()
        if view.api_key is None:
            await response.reply("Login cancelled.")
            return

        horde_api = HordeAPI(
            aiohttp.ClientSession(headers=self._common_headers | {"apikey": view.api_key}),
            logger=self.logger,
        )
        if (current_connection := self.horde_apis.get(ctx.author.id)) is not None:
            await current_connection.session.close()
        self.horde_apis[ctx.author.id] = horde_api
        try:
            await horde_api.get_current_user()  # Will fail if the user does not exist
            self.db_cursor.execute(
                # language=SQLite
                """
                INSERT OR REPLACE INTO users (discord_id, horde_api_key, civitai_api_key)
                VALUES (?, ?, ?)
                """,
                (ctx.author.id, view.api_key, None),
            )
            self.database.commit()
        except HordeRequestError:
            await response.edit(content="Invalid API key.", view=None)
            await horde_api.session.close()
            del self.horde_apis[ctx.author.id]
            return
        except sqlite3.Error:
            await response.edit(content="Internal database error occurred. Please report this to the bot owner.")
            self.logger.exception("Database error occurred while logging in.", view=None)
            return

        await response.edit(content="Successfully logged in.", view=None)

    @commands.hybrid_command()
    async def logout(self, ctx: commands.Context) -> None:
        self.db_cursor.execute(
            # language=SQLite
            """
            DELETE FROM users
            WHERE discord_id = ?
            """,
            (ctx.author.id,),
        )
        self.database.commit()
        horde_api = self.horde_apis.pop(ctx.author.id, None)
        if horde_api is not None:
            await horde_api.session.close()
        await ctx.reply("Successfully logged out.")

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
            + (f"Style: {style.name}" if random_style else ""),
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

        async for finished_image_pair in self.horde_for(ctx.author).generate_image(request):
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
            ),
        )

    @commands.hybrid_command()
    async def describe(self, ctx: commands.Context, image_url: str) -> None:
        response = await ctx.reply("Requesting interrogation... Please wait.")

        finished_interrogation = await self.horde_for(ctx.author).interrogate(InterrogationRequest(
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
        apis = APIPackage(self.horde_for(ctx.author), self.civitai, self.cache, self.logger, self.generic_session)
        view = GenerationSettingsView(apis=apis, default_request=generation_request, author_id=ctx.author.id)
        await ctx.reply(
            "Choose generation settings",
            view=view,
            embeds=await get_settings_embeds(generation_request, apis),
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
