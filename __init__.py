import contextlib
import datetime
import random
import sqlite3

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
# noinspection PyProtectedMember
from discord.user import _UserTag as UserTag

import breadcord
from .advanced_generate import DeleteOrRetryView, GenerationSettingsView, files_from_request, get_settings_embeds, \
    AttachmentDeletionView
from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion, ModelType, SortOptions, SearchPeriod
from .ai_horde.models.general import HordeRequestError
from .ai_horde.models.horde_meta import HordeNews, HordeUser, Worker
from .ai_horde.models.image import (
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
from .ai_horde.models.text import TextGenerationRequest, TextGenerationParams
from .civitai_browser import CivitAIModelBrowserView
from .helpers import APIPackage, fetch_image, report_error, map_flag_emojis, bool_emoji, format_embed_desc
from .login import LoginButtonView


# === Big to do list ===
# TODO: Allow seeing previews for models, get data from https://github.com/Haidra-Org/AI-Horde-image-model-reference
#  This can also be used to provide descriptions to models in the generation command.
#  Make sure to cache it.


class ArtificialDreaming(breadcord.module.ModuleCog):
    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        # All set in cog_load
        self.generic_session: aiohttp.ClientSession | None = None
        self.anon_horde: HordeAPI | None = None
        self.civitai: CivitAIAPI | None = None
        self.cache: Cache | None = None

        self.database: sqlite3.Connection = sqlite3.connect(self.module.storage_path / "database.db")
        self.db_cursor: sqlite3.Cursor = self.database.cursor()
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

        self._common_headers = {
            "User-Agent": f"Breadcord {self.module.manifest.name}/{self.module.manifest.version}",
        }
        self.horde_apis: dict[int, HordeAPI] = {}

        self.edit_ctx_menu = app_commands.ContextMenu(
            name="Edit image with AI",
            callback=self.edit_image_callback,
        )
        self.bot.tree.add_command(self.edit_ctx_menu)

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
            formatted_cache=bool(self.bot.settings.debug.value),
        )

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

        self.bot.add_view(AttachmentDeletionView())

    async def cog_unload(self) -> None:
        if self.generic_session is not None and not self.generic_session.closed:
            await self.generic_session.close()
        if self.civitai.session is not None and not self.civitai.session.closed:
            await self.civitai.session.close()
        if self.anon_horde is not None and not self.anon_horde.session.closed:
            await self.anon_horde.session.close()
        for api in self.horde_apis.values():
            if not api.session.closed:
                await api.session.close()

        self.update_cache.cancel()
        self.database.close()

    @tasks.loop(minutes=30)
    async def update_cache(self) -> None:
        await self.cache.update()

    @commands.hybrid_group(
        name="horde",
        description="Run commands for the AI Horde.",
    )
    async def horde(self, ctx: commands.Context) -> None:
        pass

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

    def horde_for(self, user: discord.User | discord.Member | int | UserTag | None) -> HordeAPI:
        if user is None:
            return self.anon_horde
        if isinstance(user, UserTag):
            user = user.id
        api = self.horde_apis.get(user)
        if api is None:
            self.logger.debug("Using horde anonymously")
            return self.anon_horde
        return api

    @horde.command()
    async def login(self, ctx: commands.Context) -> None:
        """Log in to the AI Horde."""
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

    @horde.command()
    async def logout(self, ctx: commands.Context) -> None:
        """Log out of the AI Horde."""
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

    @horde.command()
    @app_commands.autocomplete(
        style=style_autocomplete,
    )
    async def image_generate(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        style: str | None = None,
    ) -> None:
        """Generate an image."""
        chosen_style: Style | None = None
        if style is not None:
            chosen_style = self.style_from_name(style if style != "random" else random.choice(self.cache.styles).name)
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
            view=AttachmentDeletionView()
        )

    @horde.command()
    async def advanced_generate(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> None:
        """Generate an image with advanced customization options."""
        image_urls = [attachment.proxy_url for attachment in ctx.message.attachments]
        generation_request = ImageGenerationRequest(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            source_image=image_urls[0] if image_urls else None,
            params=ImageGenerationParams(
                karras=True,
            ),
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

    @horde.command()
    async def civitai_search(
        self,
        ctx: commands.Context,
        *,
        query: str | None = None,
        sorting: SortOptions | None = None,
        period: SearchPeriod | None = None,
        model_type: ModelType | None = None,
        show_nsfw: bool = True,  # Default to true in NSFW channels
        by_user: str | None = None,
    ) -> None:
        """Search for models (diffuser, LoRA, TI, etc) on CivitAI."""
        if ctx.interaction:
            await ctx.defer()

        if hasattr(ctx.channel, "is_nsfw"):
            show_nsfw = ctx.channel.is_nsfw() and show_nsfw  # type: ignore[attr-defined]

        models: list[CivitAIModel] | None = None
        if query:
            with contextlib.suppress(HordeRequestError):
                model = await self.civitai.get_model(query)
                models = [model] if model and (model_type is None or model.type == model_type) else None
        if not models:
            models = await self.civitai.get_models(
                query=query,
                sort=sorting,
                period=period,
                types=[model_type] if model_type else None,
                nsfw=show_nsfw,
                creator_username=by_user,
            )

        if not models:
            await ctx.reply("No models found.", ephemeral=True)
            return

        if not show_nsfw:
            models = [model for model in models if not model.nsfw]

        view = CivitAIModelBrowserView(models, cache=self.cache)
        await ctx.reply(**(await view.get_page()).unpack(), view=view)

    async def edit_image_callback(self, interaction: discord.Interaction, message: discord.Message) -> None:
        image_urls = [attachment.proxy_url for attachment in message.attachments]
        for embed in message.embeds:
            image_urls.extend(url for url in (
                embed.image.url,
                embed.thumbnail.url,
            ) if url)
        if not image_urls or not image_urls[0]:
            await interaction.response.send_message("Message does not contain an image.", ephemeral=True)
            return

        generation_request = ImageGenerationRequest(
            positive_prompt=message.content[:200] or "No prompt! Go into the basic options to set one.",
            source_image=image_urls[0],
            params=ImageGenerationParams(
                karras=True,
            ),
        )
        apis = APIPackage(self.horde_for(interaction.user), self.civitai, self.cache, self.logger, self.generic_session)
        try:
            view = GenerationSettingsView(apis=apis, default_request=generation_request, author_id=interaction.user.id)
            await interaction.response.send_message(
                "Choose generation settings",
                view=view,
                embeds=await get_settings_embeds(generation_request, apis),
                files=(await files_from_request(generation_request, session=self.generic_session))[:10],
            )
        except HordeRequestError as error:
            await report_error(interaction, error)
            self.logger.exception("Error occurred while generating image.")

    @horde.command()
    async def text_generate(self, ctx: commands.Context, *, prompt: str) -> None:
        # TODO: Figure out how to make it act like a chat, even when we don't know the model that's going to be used?
        request = TextGenerationRequest(
            prompt=prompt,
            params=TextGenerationParams(
                max_length=256,
                single_line=True,
                remove_unfinished_tail=True,
            ),
            allow_downgrade=True,
        )
        async with ctx.typing():
            generations = await anext(self.horde_for(ctx.author).generate_text(request))

        for generation in generations:
            text = f"{prompt.rstrip()} {generation.text}"
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            if len(chunks) != 1:
                await ctx.reply(chunks[0])

            for chunk in chunks[1:-1]:
                await ctx.send(chunk)

            await (ctx.reply if len(chunks) == 1 else ctx.send)(
                content=chunks[-1],
                embed=discord.Embed(
                    title="Text Generation",
                    description="\n".join((
                        f"**Model:** {generation.model}",
                        f"**Finished by:** {generation.worker_name} (`{generation.worker_id}`)",
                    )),
                ),
            )

    @horde.command(name="user")
    async def user_info(self, ctx: commands.Context, user: str | None = None) -> None:
        """Get information about a user."""
        try:
            user_id = int(user.split("#")[-1]) if user else None
        except ValueError:
            await ctx.reply(
                "Invalid user. Make sure to give the user's ID and not only their name.",
                ephemeral=True,
            )
            return
        if not user_id and self.horde_apis.get(ctx.author.id) is None:
            await ctx.reply(
                "You are not logged in. "
                "Either provide a user ID or log in with `/login` to see your own information.",
                ephemeral=True,
            )
            return
        horde_api = self.horde_for(ctx.author)

        user_data: HordeUser
        if user_id:
            try:
                user_data = await horde_api.get_user(user_id)
            except HordeRequestError as error:
                await ctx.reply(
                    f"User with ID `{user_id}` not found."
                    if error.code == 404 else
                    f"Unable to get data for the user with ID `{user_id}`.",
                    ephemeral=True,
                )
                return
        else:
            user_data = await horde_api.get_current_user()

        embed = discord.Embed(
            title=f"User {user_data.username.removesuffix(f'#{user_data.id}')}",
            description=format_embed_desc({
                "ID": f"`{user_data.id}`",
                "Moderator": bool_emoji(True) if user_data.moderator else None,
                "Service": bool_emoji(True) if user_data.service else None,
                "Trusted": bool_emoji(user_data.trusted),
                "Account created": f"<t:{int((datetime.datetime.now() - user_data.account_age).timestamp())}:R>",
                "Max concurrent generations": user_data.concurrency,
                "Workers": user_data.worker_count,
            }),
        )
        if user_data.worker_ids:
            embed.add_field(
                name="Workers",
                value="\n".join([
                    f"{discord.utils.escape_markdown(worker.name)} `{worker_id}` " + " ".join(map_flag_emojis(
                        (worker.online, (":satellite:", ":no_entry_sign:")),
                        (worker.trusted, ":shield:"),
                        (worker.flagged, ":triangular_flag_on_post:"),
                        (worker.maintenance_mode, ":pause_button:"),
                    ))
                    for worker_id in user_data.worker_ids
                    for worker in [await horde_api.get_worker(worker_id)]
                ]),
                inline=False,
            )
        embed.add_field(
            name="Contributions",
            value=format_embed_desc({
                "Images": f"{user_data.records.fulfillment.image}"
                          f" ({user_data.records.contribution.megapixelsteps:.0f} MPS)",
                "Text": f"{user_data.records.fulfillment.text}"
                        f" ({user_data.records.contribution.tokens} tokens)",
                "Interrogations": f"{user_data.records.fulfillment.interrogation}",
            }),
            inline=False,
        )
        embed.add_field(
            name="Usage",
            value=format_embed_desc({
                "Images": f"{user_data.records.request.image}"
                          f" ({user_data.records.usage.megapixelsteps:.0f} MPS)",
                "Text": f"{user_data.records.request.text}"
                        f" ({user_data.records.usage.tokens} tokens)",
                "Interrogations": f"{user_data.records.request.interrogation}",
            }),
            inline=False,
        )

        await ctx.reply(embed=embed)

    @horde.command(name="worker")
    async def worker_info(self, ctx: commands.Context, worker_id: str) -> None:
        """Get information about a worker."""
        try:
            worker = await self.horde_for(ctx.author).get_worker(worker_id.strip())
        except HordeRequestError as error:
            await ctx.reply(
                f"Worker with ID `{worker_id}` not found."
                if error.code == 404 else
                f"Unable to get data for the worker with ID `{worker_id}`.",
                ephemeral=True,
            )
            return

        flag_emojis = " ".join(map_flag_emojis(
            (worker.online, (":satellite:", ":no_entry_sign:")),
            (worker.trusted, ":shield:"),
            (worker.flagged, ":triangular_flag_on_post:"),
            (worker.maintenance_mode, ":pause_button:"),
        ))
        embed = discord.Embed(
            title=f"Worker `{worker.name}` {flag_emojis}",
            description=format_embed_desc({
                "ID": f"`{worker.id}`",
                "Online": bool_emoji(worker.online),
                "Trusted": bool_emoji(worker.trusted),
                "Flagged": bool_emoji(True) if worker.flagged else None,
                "Maintenance mode": bool_emoji(True) if worker.maintenance_mode else None,
                "Worker Type": worker.type.name.capitalize(),
                "Models": ", ".join((f"`{model}`" for model in sorted(worker.models or []))) or None,
                "Forms": ", ".join((f"`{form}`" for form in sorted(worker.forms or []))) or None,
            }),
        ).add_field(
            name="Stats",
            value=format_embed_desc({
                "Total uptime": readable_delta(worker.uptime),
                "Kudos received": f"{round(worker.kudos_rewards):,}",
                "Performance": worker.performance,
                "Requests received": f"{worker.requests_fulfilled} (+{worker.uncompleted_jobs} uncompleted)",
                "Megapixelsteps generated": f"{round(worker.megapixelsteps_generated):,} MPS",
                "Text tokens generated": f"{worker.tokens_generated:,}",
            }),
        ).add_field(
            name="Generation info",
            value=format_embed_desc({
                "Threads": worker.threads,
                "Max resolution": f"{(width := round(worker.max_pixels ** 0.5) // 64 * 64)}x{width}"
                                  f" ({worker.max_pixels:,} pixels)" if worker.max_pixels else None,
                "Max tokens": f"{worker.max_length}"
                              + (f" ({worker.max_context_length} context)" if worker.max_context_length else "")
                              if worker.max_length else None,
                "": "",
                "Image-to-image": bool_emoji(worker.img2img),
                "Inpainting": bool_emoji(worker.painting),
                "LoRAs": bool_emoji(worker.lora),
                "Post-processing": bool_emoji(worker.post_processing),
            }),
            inline=False,
        )
        if worker.team:
            team = await self.horde_for(ctx.author).get_team(worker.team.id)
            embed.add_field(
                name="Part of Team",
                value=format_embed_desc({
                    "Name": team.name,
                    "ID": f"`{team.id}`",
                }),
                inline=False,
            )

        embed.set_footer(text=worker.bridge_agent)
        await ctx.reply(embed=embed)


async def setup(bot: breadcord.Bot):
    await bot.add_cog(ArtificialDreaming("artificial_dreaming"))
