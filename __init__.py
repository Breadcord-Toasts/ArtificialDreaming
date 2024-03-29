import contextlib
import random
import sqlite3

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
# noinspection PyProtectedMember
from discord.user import _UserTag as UserTag

import breadcord
from .advanced_generate import GenerationSettingsView, get_settings_embeds
from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI, SearchCategory
from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion, ModelType, SearchFilter, SortOptions
from .ai_horde.models.general import HordeRequestError
from .ai_horde.models.horde_meta import HordeNews
from .ai_horde.models.image import (
    Base64Image,
    CaptionResult,
    ExtraSourceImage,
    GenericProcessedImageResult,
    ImageGenerationParams,
    ImageGenerationRequest,
    InterrogationRequest,
    InterrogationRequestForm,
    InterrogationType,
    LoRA,
    Sampler,
    SourceProcessing,
    TextualInversion,
)
from .ai_horde.models.other_sources import Style
from .ai_horde.models.text import TextGenerationRequest
from .civitai_browser import CivitAIModelBrowserView
from .helpers import APIPackage, LongLastingView, fetch_image, report_error
from .login import LoginButtonView


class NoneWithProperties:
    """Provides a nicer interface for accessing properties of objects that may not be defined."""

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, item) -> "NoneWithProperties":
        return self

    def __getitem__(self, item) -> "NoneWithProperties":
        return self


# === Big to do list ===
# TODO: Allow seeing previews for models, get data from https://github.com/Haidra-Org/AI-Horde-image-model-reference
#  This can also be used to provide descriptions to models in the generation command.
#  Make sure to cache it.


class ArtificialDreaming(
    breadcord.module.ModuleCog,
    commands.GroupCog,
    group_name="horde",
    group_description="Run commands for the AI Horde.",
):
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

    @tasks.loop(minutes=30)
    async def update_cache(self) -> None:
        await self.cache.update()

    async def style_autocomplete(self, _, value: str) -> list[app_commands.Choice[str]]:
        styles = [style.name for style in self.cache.styles] + list(self.cache.style_categories.keys())

        if not value:
            return [
                app_commands.Choice(name="Random style", value="random"),
            ] + [
                app_commands.Choice(name=style, value=style)
                for style in random.sample(styles, 24)
            ]

        # noinspection PyArgumentEqualDefault
        return [
            app_commands.Choice(name=style, value=style)
            for style in breadcord.helpers.search_for(
                query=value,
                objects=styles,
                max_results=25,
            )
        ]

    def style_from_name(self, name: str) -> Style | None:
        def to_style(style_name: str) -> Style | None:
            for style in self.cache.styles:
                if style.name.lower() == style_name:
                    return style
            return None

        def from_category(category_name: str) -> Style | None:
            category = self.cache.style_categories.get(category_name)
            if category is not None:
                style_name: str = random.choice(category)
                return to_style(style_name)
            return None

        name = name.lower().strip()
        return to_style(name) or from_category(name)

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
                "If you do not have an account, you can make one [here](https://aihorde.net/register)."
                "\n\n"
                "### **Important!** \n"
                "Your API key will be used to perform actions on your behalf. "
                "If you do not trust the bot owner with these permissions, consider using a shared token."
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
    @app_commands.autocomplete(
        style=style_autocomplete,
    )
    async def generate(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        style: str | None = None,
    ) -> None:
        chosen_style: Style | None = None
        if style is not None:
            chosen_style = self.style_from_name(style)
            if chosen_style is None:
                raise commands.BadArgument("Invalid style.")

        response = await ctx.reply(
            "Generating image... Please wait. \n"
            + (f"Style: `{chosen_style.name}`" if chosen_style is not None else ""),
        )

        if chosen_style is not None:
            request = ImageGenerationRequest(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                nsfw=True,
                params=ImageGenerationParams(image_count=1),
                replacement_filter=True,
            ).apply_style(chosen_style, cache=self.cache)
        else:
            request = ImageGenerationRequest(
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                nsfw=True,
                models=["AlbedoBase XL (SDXL)", "Fustercluck", "ICBINP XL"],
                params=ImageGenerationParams(
                    width=1024,
                    height=1024,
                    sampler=Sampler.K_EULER_A,
                    loras=[LoRA(identifier="246747", is_version=True)],
                    steps=8,
                    cfg_scale=2.0,
                    image_count=1,
                ),
                allow_downgrade=True,
                replacement_filter=True,
            )
        try:
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
        except HordeRequestError as error:
            await report_error(ctx, error)
            self.logger.exception("Error occurred while generating image.")
            return

        await response.edit(
            content=(
                "Finished generation. \n"
                + (f"Style: {chosen_style.name}" if chosen_style is not None else "")
            ),
        )

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
        try:
            view = GenerationSettingsView(apis=apis, default_request=generation_request, author_id=ctx.author.id)
            await ctx.reply(
                "Choose generation settings",
                view=view,
                embeds=await get_settings_embeds(generation_request, apis),
            )
        except HordeRequestError as error:
            await report_error(ctx, error)
            self.logger.exception("Error occurred while generating image.")

    @commands.hybrid_command()
    async def civitai_search(
        self,
        ctx: commands.Context,
        *, query: str | None = None,
        sorting: SortOptions | None = None,
        model_type: ModelType | None = None,
        show_nsfw: bool = False,
    ) -> None:
        if ctx.interaction:
            await ctx.defer()

        models: list[CivitAIModel] | None = None
        if not query:
            models = await self.civitai.get_models(type=model_type)
        if not models:
            with contextlib.suppress(HordeRequestError):
                model = await self.civitai.get_model(query)
                models = [model] if model and (model_type is None or model.type == model_type) else None
        if not models:
            with contextlib.suppress(HordeRequestError):
                models = await self.civitai.search(
                    query,
                    SearchCategory.MODELS,
                    filters=SearchFilter().model_type(model_type) if model_type else None,
                    sort=sorting,
                    limit=10,
                )

        if not models:
            await ctx.reply("No models found.", ephemeral=True)
            return

        if not show_nsfw:
            models = [model for model in models if not model.nsfw]

        view = CivitAIModelBrowserView(models, cache=self.cache)
        await ctx.reply(**(await view.get_page()).unpack(), view=view)


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
