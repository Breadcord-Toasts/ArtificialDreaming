import random
import sqlite3
from typing import TYPE_CHECKING

import aiohttp
import discord
from discord import app_commands
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
from .ai_horde.models.other_sources import Style
from .ai_horde.models.text import TextGenerationRequest
from .helpers import APIPackage, fetch_image
from .login import LoginButtonView

if TYPE_CHECKING:
    from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion


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

        def from_category(category_name: str) -> Style | None:
            category = self.cache.style_categories.get(category_name)
            if category is not None:
                style_name: str = random.choice(category)
                return to_style(style_name)

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
            await response.edit(content=f"Error occurred while generating image: {error}")
        else:
            await response.edit(
                content=(
                    "Finished generation. \n"
                    + (f"Style: {chosen_style.name}" if chosen_style is not None else "")
                ),
            )

    @commands.hybrid_command()
    async def describe(self, ctx: commands.Context, image_url: str) -> None:
        response = await ctx.reply("Requesting interrogation... Please wait.")
        try:
            finished_interrogation = await self.horde_for(ctx.author).interrogate(InterrogationRequest(
                image_url=image_url,
                forms=[InterrogationRequestForm(name=InterrogationType.CAPTION)],
            ))
            result: CaptionResult = finished_interrogation.forms[0].result
        except HordeRequestError as error:
            await response.edit(content=f"Error occurred while interrogating image: {error}")
        else:
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
        try:
            view = GenerationSettingsView(apis=apis, default_request=generation_request, author_id=ctx.author.id)
            await ctx.reply(
                "Choose generation settings",
                view=view,
                embeds=await get_settings_embeds(generation_request, apis),
            )
        except HordeRequestError as error:
            await ctx.send(f"Encountered an error from the AI Horde: {error}")

    @commands.hybrid_command(description="Get info about a model from CivitAI.")
    @app_commands.describe(model_id="The ID as shown in the CivitAI URL.")
    async def civitai_model(self, ctx: commands.Context, model_id: int) -> None:
        # TODO: Integrate with horde model reference?
        model: CivitAIModel | None = None
        version: CivitAIModelVersion | NoneWithProperties = NoneWithProperties()
        try:
            model = await self.civitai.get_model(model_id)
        except HordeRequestError:
            version = await self.civitai.get_model_version(model_id)
            if version is not None:
                model = await self.civitai.get_model(version.model_id)
        finally:
            if model is None:
                await ctx.reply("Model not found.", ephemeral=True)
                return

        def hacky_camel_case_split(string: str) -> str:
            if len(string) < 8:  # VERY hacky, but oh well
                return string
            out = ""
            for i in range(len(string)):
                char = string[i]
                out += char
                next_char = string[nxt] if (nxt := i+1) < len(string) else ""
                if char.islower() and next_char.isupper():
                    out += " "
            return out

        embed = discord.Embed(
            title=version.name or model.name,
            url=version.url or model.url,
            colour=discord.Colour.random(seed=model.id),
            description="\n".join(s for s in (
                f"Version of [{model.name}]({model.url})" if version else "",
                f"**Model type:** {hacky_camel_case_split(model.type.value)}",
                f"**NSFW:** {model.nsfw}" if model.nsfw else "",
            ) if s),
        )
        embed.set_image(url=version.sfw_thumbnail_url or model.sfw_thumbnail_url)
        embed.set_author(
            name=model.creator.username,
            icon_url=model.creator.image_url,
        )
        footer = f"Model ID: {model.id}"
        if version:
            footer += f" | Version ID: {version.id}"
        embed.set_footer(text=footer)
        embed.add_field(
            name=((stats := model.stats) and False) or "Stats",
            value="\n".join(s for s in (
                f"**Downloads:** {stats.download_count:,}",
                f"**Rating:** {stats.rating} ({stats.ratingCount:,} ratings)" if stats.rating is not None else "",
            ) if s),
            inline=False,
        )
        file = (version or model.versions[0]).files[0]
        if file.size_kb > 1024 ** 2:
            appropriate_filesize = f"{file.size_kb / (1024 * 1024):.2f} GB"
        elif file.size_kb > 1024:
            appropriate_filesize = f"{file.size_kb / 1024:.2f} MB"
        else:
            appropriate_filesize = f"{file.size_kb:.2f} KB"
        embed.add_field(
            name="File",
            value="\n".join(s for s in (
                f"[{file.name}]({file.download_url})",
                f"**Size:** {appropriate_filesize}",
                f"**Type:** {hacky_camel_case_split(file.type.value)}",
            ) if s),
            inline=False,
        )

        await ctx.reply(embed=embed)


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
