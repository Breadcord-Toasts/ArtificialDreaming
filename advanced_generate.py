import math
from logging import Logger
from typing import TYPE_CHECKING, Any, NamedTuple

import discord
from discord import Interaction

from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.horde_meta import ActiveModel
from .ai_horde.models.image import (
    ImageGenerationParams,
    ImageGenerationRequest,
    LoRA,
    TextualInversion,
    TIPlacement,
    Sampler
)

if TYPE_CHECKING:
    from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion


class APIPackage(NamedTuple):
    horde: HordeAPI
    civitai: CivitAIAPI
    cache: Cache


class CustomModelModal(discord.ui.Modal, title="Custom model"):
    def __init__(self, *args, possible_models: list[ActiveModel], **kwargs):
        super().__init__(*args, **kwargs)
        self.models: list[str] | None = None
        self._possible_models = possible_models

    models_input = discord.ui.TextInput(
        label="Models",
        placeholder="Enter a comma-separated list of model names...",
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        models = [
            model.strip()
            for model in self.models_input.value.split(",")
            if model.strip()
        ]
        possible_models = tuple(model.name for model in self._possible_models)
        for model in models:
            if model not in possible_models:
                await interaction.response.send_message(
                    (
                        f"Model {model!r} is not available. "
                        f"Models are matched exactly, "
                        f"so make sure that it is spelled and capitalised correctly, and that the model is available."
                    ),
                    ephemeral=True,
                )
                return

        self.models = models
        await interaction.response.defer()


class ModelSelect(discord.ui.Select):
    def __init__(self, available_models: list[ActiveModel], *, modify: ImageGenerationRequest, apis: APIPackage):
        self.to_modify = modify
        self.apis = apis

        available_models = sorted(available_models, key=lambda model: model.queued, reverse=True)
        self.available_models = [
            model
            for model in available_models
            if model.count > 0 and model.type == "image"
        ]

        super().__init__(
            placeholder="Select a model...",
            min_values=0,
            max_values=25,
            options=[
                discord.SelectOption(
                    label="Any",
                    value="UNSET",
                    description="Allows any model to generate this request.",
                ),
                discord.SelectOption(
                    label="Custom",
                    value="CUSTOM",
                    description="Allows you to specify a custom list of models.",
                ),
            ] + [
                discord.SelectOption(
                    label=model.name,
                    description=f"Count: {model.count} | Queued: {model.queued}",
                )
                for model in self.available_models[:23]
            ],
        )

    async def callback(self, interaction: discord.Interaction):
        unset = len(self.values) == 1 and self.options[0].value == "UNSET"
        if "UNSET" in self.values:
            self.values.remove("UNSET")
        if "CUSTOM" in self.values:
            modal = CustomModelModal(possible_models=self.available_models)
            await interaction.response.send_modal(modal)
            await modal.wait()
            self.to_modify.models = modal.models
            await defer_and_edit(interaction, self.to_modify, self.apis, responded_already=True)
            return

        self.to_modify.models = None if unset else self.values
        await defer_and_edit(interaction, self.to_modify, self.apis)


class CustomResolutionModal(discord.ui.Modal, title="Custom resolution"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width: int | None = None
        self.height: int | None = None

    width_input = discord.ui.TextInput(
        label="Width",
        placeholder="Enter a width...",
    )
    height_input = discord.ui.TextInput(
        label="Height",
        placeholder="Enter a height...",
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        width = round(int(self.width_input.value) / 64) * 64
        width = max(64, min(3072, width))
        self.width = width

        height = round(int(self.height_input.value) / 64) * 64
        height = max(64, min(3072, height))
        self.height = height

        await interaction.response.defer()


class ResolutionSelect(discord.ui.Select):
    def __init__(self, base_res: int = 512, *, modify: ImageGenerationRequest, apis: APIPackage):
        self.to_modify = modify
        self.apis = apis

        base_options: dict[str, tuple[float, float] | None] = {
            # Normal (512 base)
            "Square": (base_res, base_res),
            "Portrait": (base_res, base_res * 1.5),
            "Landscape": (base_res * 1.5, base_res),
            # Large (1024 base)
            "Square (Large)": (base_res * 2, base_res * 2),
            "Portrait (Large)": (base_res * 2, base_res * 3),
            "Landscape (Large)": (base_res * 3, base_res * 2),

            "Custom": None,
        }

        options = []
        for name, value in base_options.items():
            if value is None:
                options.append(discord.SelectOption(
                    label=name,
                    description="A custom resolution...",
                    value="Custom",
                ))
                continue
            width, height = value
            # Round to nearest 64 multiple
            width = round(width / 64) * 64
            height = round(height / 64) * 64
            gcd = math.gcd(width, height)

            options.append(discord.SelectOption(
                label=name,
                description=f"{width}x{height}: {width // gcd}:{height // gcd}",
                value=f"{width}x{height}",
            ))

        super().__init__(placeholder="Select a resolution...", options=options)

    async def callback(self, interaction: discord.Interaction):
        # This would be much cleaner with goto ;)
        async def inner() -> bool:
            if self.values[0] == "Custom":
                modal = CustomResolutionModal()
                await interaction.response.send_modal(modal)
                responded_already = True
                await modal.wait()
                width, height = modal.width, modal.height
                if width is None or height is None:
                    return responded_already
            else:
                width, height = self.values[0].split("x")
                responded_already = False
            self.to_modify.params.width = int(width)
            self.to_modify.params.height = int(height)
            return responded_already

        await defer_and_edit(interaction, self.to_modify, self.apis, responded_already=await inner())


class BasicOptionsModal(discord.ui.Modal, title="More generation options"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

        self.positive_prompt = discord.ui.TextInput(
            label="Prompt",
            placeholder="Enter a prompt...",
            style=discord.TextStyle.paragraph,
            min_length=1,
            # Max total prompt length (positive + negative) is 1000, we do half - 50 in case something else is injected
            max_length=450,
            default=self.current_request.positive_prompt,
        )
        self.add_item(self.positive_prompt)

        self.negative_prompt = discord.ui.TextInput(
            label="Negative prompt",
            placeholder="Enter a negative prompt...",
            style=discord.TextStyle.paragraph,
            required=False,
            # Max total prompt length (positive + negative) is 1000, we do half - 50 in case something else is injected
            max_length=450,
            default=self.current_request.negative_prompt,
        )
        self.add_item(self.negative_prompt)

        self.seed = discord.ui.TextInput(
            label="Seed",
            placeholder="Enter a seed...",
            required=False,
            default=self.current_request.params.seed,
        )
        self.add_item(self.seed)

        self.cfg_scale = discord.ui.TextInput(
            label="CFG scale - 0 \u2264 cfg \u2264 100",
            placeholder="Enter a CFG scale...",
            required=False,
            default=self.current_request.params.cfg_scale,
        )
        self.add_item(self.cfg_scale)

        self.steps = discord.ui.TextInput(
            label="Steps - 1 \u2264 steps \u2264 100",
            placeholder="Enter a step count...",
            required=False,
            default=self.current_request.params.steps,
        )
        self.add_item(self.steps)

        self.image_count = discord.ui.TextInput(
            label="Image count - 1 \u2264 images \u2264 100",
            placeholder="Enter an image count...",
            required=False,
            default=self.current_request.params.image_count,
        )
        self.add_item(self.image_count)

    async def on_submit(self, interaction: Interaction, /) -> None:
        self.current_request.positive_prompt = self.positive_prompt.value
        self.current_request.negative_prompt = self.negative_prompt.value or None
        self.current_request.params.seed = self.seed.value or None

        cfg_scale = self.cfg_scale.value.strip()
        cfg_scale = None if cfg_scale else max(0.0, min(100.0, float(cfg_scale)))
        self.current_request.params.cfg_scale = cfg_scale

        steps = self.steps.value.strip()
        steps = None if steps else max(1, min(100, int(steps)))
        self.current_request.params.steps = steps

        image_count = self.image_count.value.strip()
        image_count = None if image_count else max(1, min(20, int(image_count)))
        self.current_request.params.image_count = image_count

        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class AdvancedOptionsModal(discord.ui.Modal, title="More generation options"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

        self.sampler = discord.ui.TextInput(
            label="Sampler",
            placeholder="Enter a sampler...",
            required=False,
            default=self.current_request.params.sampler,
        )
        self.add_item(self.sampler)

        self.denoising_strength = discord.ui.TextInput(
            label="Denoising strength - 0 \u2264 strength \u2264 100",
            placeholder="Enter a denoising strength...",
            required=False,
            default=self.current_request.params.denoising_strength,
        )
        self.add_item(self.denoising_strength)

    async def on_submit(self, interaction: Interaction, /) -> None:
        if (key := self.sampler.value.strip().upper()) in Sampler.__members__:
            sampler = Sampler[key]
        else:
            sampler = self.sampler.value.strip()
        self.current_request.params.sampler = sampler

        denoising_strength = self.denoising_strength.value.strip()
        denoising_strength = None if denoising_strength else max(0.01, min(1.0, float(denoising_strength)))
        self.current_request.params.denoising_strength = denoising_strength

        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)
        raise error


class LoRAPickerModal(discord.ui.Modal, title="LoRA"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

    identifier = discord.ui.TextInput(
        label="LoRA ID or name",
        placeholder="Enter a CivitAI LoRA model or LoRA version ID, or its exact name...",
        max_length=255,
    )
    model_strength = discord.ui.TextInput(
        label="LoRA model strength - -5 \u2264 strength \u2264 5",
        placeholder="Enter a model strength...",
        required=False,
    )
    clip_strength = discord.ui.TextInput(
        label="LoRA clip strength - -5 \u2264 strength \u2264 5",
        placeholder="Enter a clip strength...",
        required=False,
    )
    is_version = discord.ui.TextInput(
        label="If the LoRA ID is a version ID - True/False",
        placeholder='Enter "True" or "False"...',
        required=False,
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        self.current_request.params.loras = self.current_request.params.loras or []
        self.current_request.params.loras.append(LoRA(
            identifier=self.identifier.value.strip(),
            strength_model=float(self.model_strength.value) if self.model_strength.value else None,
            strength_clip=float(self.clip_strength.value) if self.clip_strength.value else None,
            is_version=check_truthy(self.is_version.value) if self.is_version.value else None,
        ))
        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class TextualInversionPickerModal(discord.ui.Modal, title="Textual Inversion"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

    identifier = discord.ui.TextInput(
        label="Textual Inversion ID or name",
        placeholder="Enter a CivitAI Textual Inversion model ID, or its exact name...",
        max_length=255,
    )
    injection_location = discord.ui.TextInput(
        label="Injection location (prompt/negative prompt)",
        placeholder='Enter "prompt" or "negative prompt"...',
        default="Prompt",
    )
    strength = discord.ui.TextInput(
        label="Textual Inversion strength - -5 \u2264 str \u2264 5",
        placeholder="Enter a strength...",
        required=False,
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        def transform_location(value: str) -> TIPlacement | None:
            value = value.strip().lower()
            if value.startswith("p"):
                return TIPlacement.PROMPT
            if value.startswith("n"):
                return TIPlacement.NEGATIVE_PROMPT
            return None

        self.current_request.params.textual_inversions = self.current_request.params.textual_inversions or []
        self.current_request.params.textual_inversions.append(TextualInversion(
            identifier=self.identifier.value.strip(),
            injection_location=transform_location(self.injection_location.value),
            strength=float(self.strength.value) if self.strength.value else None,
        ))
        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class GenerationSettingsView(discord.ui.View):
    def __init__(
        self,
        logger: Logger,
        cache: Cache,
        horde_api: HordeAPI,
        civitai_api: CivitAIAPI,
        default_request: ImageGenerationRequest,
        author_id: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.author_id = author_id
        self.logger = logger
        self.cache = cache
        self.horde_api = horde_api
        self.civitai_api = civitai_api
        self.apis = APIPackage(
            horde=horde_api,
            civitai=civitai_api,
            cache=cache,
        )

        self._default_request = default_request
        self.generation_request = default_request
        self.generation_request.params = self.generation_request.params or ImageGenerationParams()

        self.model_select = ModelSelect(self.cache.horde_models, modify=self.generation_request, apis=self.apis)
        self.add_item(self.model_select)

        self.resolution_select = ResolutionSelect(modify=self.generation_request, apis=self.apis)
        self.add_item(self.resolution_select)

        # TODO: Put support for post processors somewhere?
        #  Might need to split out lora/ti adding to another message, and also put post processors there.
        #  Remember to deal with facefixer_strength?

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.author_id

    @discord.ui.button(label="Add LoRA", style=discord.ButtonStyle.green, row=3)
    async def add_lora(self, interaction: discord.Interaction, _):
        modal = LoRAPickerModal(current_request=self.generation_request, apis=self.apis)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Remove LoRA", style=discord.ButtonStyle.red, row=3)
    async def remove_lora(self, interaction: discord.Interaction, _):
        if self.generation_request.params.loras:
            self.generation_request.params.loras.pop()
        await defer_and_edit(interaction, self.generation_request, self.apis)

    @discord.ui.button(label="Add Textual Inversion", style=discord.ButtonStyle.green, row=3)
    async def add_ti(self, interaction: discord.Interaction, _):
        modal = TextualInversionPickerModal(current_request=self.generation_request, apis=self.apis)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Remove Textual Inversion", style=discord.ButtonStyle.red, row=3)
    async def remove_ti(self, interaction: discord.Interaction, _):
        if self.generation_request.params.textual_inversions:
            self.generation_request.params.textual_inversions.pop()
        await defer_and_edit(interaction, self.generation_request, self.apis)

    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4)
    async def generate(self, interaction: discord.Interaction, _):
        for item in self.children:
            if hasattr(item, "disabled"):
                item.disabled = True
            if hasattr(item, "enabled"):
                item.enabled = False

        # TODO: Stop and return this interaction for a followup generation response?
        #  Also remove children instead of disabling them
        await defer_and_edit(interaction, self.generation_request, self.apis)

    @discord.ui.button(label="Basic options", style=discord.ButtonStyle.blurple, row=4)
    async def more_options(self, interaction: discord.Interaction, _):
        await interaction.response.send_modal(
            BasicOptionsModal(current_request=self.generation_request, apis=self.apis)
        )

    @discord.ui.button(label="Advanced options", style=discord.ButtonStyle.blurple, row=4)
    async def even_more_options(self, interaction: discord.Interaction, _):
        await interaction.response.send_modal(
            AdvancedOptionsModal(current_request=self.generation_request, apis=self.apis)
        )

    @discord.ui.button(label="Allow NSFW: No", style=discord.ButtonStyle.red, row=4)
    async def nsfw_toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.generation_request.nsfw = button.style != discord.ButtonStyle.green
        button.style = discord.ButtonStyle.green if self.generation_request.nsfw else discord.ButtonStyle.red
        button.label = f"Allow NSFW: {'Yes' if self.generation_request.nsfw else 'No'}"
        await interaction.response.defer()
        await interaction.message.edit(view=self, embeds=await embed_from_request(self.generation_request, self.apis))

    @discord.ui.button(label="Get JSON", style=discord.ButtonStyle.gray, row=4, emoji="\N{PAGE FACING UP}")
    async def get_json(self, interaction: discord.Interaction, _):
        json = self.generation_request.model_dump_json(indent=4, exclude_none=True)
        await interaction.response.send_message(
            f"AI horde request data: ```json\n{json}\n```",
            ephemeral=True,
        )
        # TODO: Allow inputting arbitrary json (through a new message asking for a reply?), and have it validated


async def defer_and_edit(
    interaction: discord.Interaction,
    generation_request: ImageGenerationRequest,
    apis: APIPackage,
    *,
    responded_already: bool = False,
) -> None:
    if not responded_already:
        await interaction.response.defer()
    await interaction.message.edit(embeds=await embed_from_request(generation_request, apis))


async def embed_from_request(generation_request: ImageGenerationRequest, apis: APIPackage,) -> list[discord.Embed]:
    description = []

    def append_truthy(key_name: str, value: Any):
        if value:
            description.append(f"**{key_name}:** {value}")

    append_truthy("Prompt", generation_request.positive_prompt)
    append_truthy("Negative prompt", generation_request.negative_prompt)
    append_truthy("Seed", generation_request.params.seed)
    append_truthy("Models", ", ".join(generation_request.models or ["Any"]))
    append_truthy(
        "Resolution",
        f"{generation_request.params.width or 512}x{generation_request.params.height or 512}",
    )
    append_truthy("Steps", generation_request.params.steps)
    append_truthy("CFG scale", generation_request.params.cfg_scale)
    append_truthy("Image count", generation_request.params.image_count or 1)
    append_truthy("Denoising strength", generation_request.params.denoising_strength)

    embeds = [
        discord.Embed(
            title="Image generation",
            description="\n".join(description),
            color=discord.Color.blurple(),
        ).set_footer(text="Click the buttons below to modify the request."),
    ]

    for lora in generation_request.params.loras or []:
        civitai_model: CivitAIModel | CivitAIModelVersion | None
        if lora.is_version:
            civitai_model = await apis.civitai.get_model_version(lora.identifier)
            images = civitai_model.images if civitai_model else []
        else:
            civitai_model = await apis.civitai.get_model(lora.identifier)
            images = civitai_model.versions[0].images if civitai_model.versions else []

        if civitai_model:
            embeds.append(
                discord.Embed(
                    title=f"LoRA: {civitai_model.name}",
                    description="\n".join((
                        f"**ID:** {civitai_model.id} {'(version)' if lora.is_version else ''}",
                        f"**Strength (model):** {lora.strength_model}",
                        f"**Strength (clip):** {lora.strength_clip}",
                    )),
                ).set_thumbnail(url=next((image.url for image in images if not image.nsfw), None)),
            )
        else:
            embeds.append(
                discord.Embed(
                    title=f"LoRA: {lora.identifier}",
                    description="\n".join((
                        f"**Strength (model):** {lora.strength_model}",
                        f"**Strength (clip):** {lora.strength_clip}",
                    )),
                ),
            )

    for ti in generation_request.params.textual_inversions or []:
        civitai_model = await apis.civitai.get_model(ti.identifier)
        if civitai_model:
            embeds.append(
                discord.Embed(
                    title=f"Textual inversion: {civitai_model.name}",
                    description="\n".join((
                        f"**ID:** {civitai_model.id}",
                        f"**Strength:** {ti.strength}",
                        f"**Injection location:** {ti.injection_location}",
                    )),
                ).set_thumbnail(url=civitai_model.versions[0].images[0].url),
            )
        else:
            embeds.append(
                discord.Embed(
                    title=f"Textual inversion: {ti.identifier}",
                    description="\n".join((
                        f"**Strength:** {ti.strength}",
                        f"**Injection location:** {ti.injection_location}",
                    )),
                ),
            )

    return embeds


def check_truthy(value: str) -> bool | None:
    value = value.strip().lower()
    if value.startswith(("t", "y")):
        return True
    if value.startswith(("f", "n")):
        return False
    return None
