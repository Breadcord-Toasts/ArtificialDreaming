from __future__ import annotations

import io
from base64 import b64decode, b64encode
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

from pydantic import (
    BeforeValidator,
    Field,
    StringConstraints,
    computed_field,
    conlist,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from .general import HordeModel, HordeRequest, HordeSuccess, RenamedField
from .horde_meta import GenerationCheck

if TYPE_CHECKING:
    from ..cache import Cache
    from .other_sources import Style

_T = TypeVar("_T")


def unique_list_validator(value: list[_T]) -> list[_T]:
    assert len(value) == len(set(value)), "List contains duplicate elements"
    return value


UniqueList = Annotated[list[_T], BeforeValidator(unique_list_validator)]


class Base64Image:
    """A Base64-encoded image."""

    def __init__(self, data: str | bytes) -> None:
        if isinstance(data, str):
            data = b64decode(data).decode()
        self.data: bytes = data

    def __bytes__(self) -> bytes:
        return self.data

    def __str__(self) -> str:
        return b64encode(self.data).decode()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def to_bytesio(self) -> io.BytesIO:
        return io.BytesIO(bytes(self))

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], _) -> CoreSchema:
        assert source is Base64Image
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )


# region Image generation
class SourceProcessing(StrEnum):
    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    REMIX = "remix"


# noinspection SpellCheckingInspection
class Sampler(StrEnum):
    DDIM = "DDIM"
    K_EULER_A = "k_euler_a"
    K_DPM_ADAPTIVE = "k_dpm_adaptive"
    K_HEUN = "k_heun"
    K_DPM_2 = "k_dpm_2"
    K_DPMPP_SDE = "k_dpmpp_sde"
    K_LMS = "k_lms"
    DPMSOLVER = "dpmsolver"
    K_DPM_FAST = "k_dpm_fast"
    K_EULER = "k_euler"
    K_DPM_2_A = "k_dpm_2_a"
    K_DPMPP_2S_A = "k_dpmpp_2s_a"
    K_DPMPP_2M = "k_dpmpp_2m"
    LCM = "lcm"


# noinspection SpellCheckingInspection
class PostProcessor(StrEnum):
    GFPGAN = "GFPGAN"
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    NMKD_SIAX = "NMKD_Siax"
    FOURX_ANIMESHARP = "4x_AnimeSharp"
    CODEFORMERS = "CodeFormers"
    STRIP_BACKGROUND = "strip_background"


# noinspection SpellCheckingInspection
class ControlType(StrEnum):
    CANNY = "canny"
    HED = "hed"
    DEPTH = "depth"
    NORMAL = "normal"
    OPENPOSE = "openpose"
    SEG = "seg"
    SCRIBBLE = "scribble"
    FAKESCRIBBLES = "fakescribbles"
    HOUGH = "hough"


# noinspection SpellCheckingInspection
class TIPlacement(StrEnum):
    PROMPT = "prompt"
    NEGATIVE_PROMPT = "negprompt"


class LoRA(HordeModel):
    identifier: str = RenamedField(
        description="The exact name or CivitAI ID of the LoRA.",
        renamed_to="identifier", original_name="name",
    )
    strength_model: float | None = RenamedField(
        default=None,
        description="The strength with which to apply the LoRA to the image generation model.",
        renamed_to="strength_model", original_name="model",
        ge=-5, le=5,
    )
    strength_clip: float | None = RenamedField(
        default=None,
        description="The strength with which to apply the LoRA to the CLIP language model.",
        renamed_to="strength_clip", original_name="clip",
        ge=-5, le=5,
    )
    # noinspection PyTypeHints
    inject_trigger: Annotated[Literal["any"] | str, StringConstraints(min_length=1, max_length=30)] | None = Field(
        default="any",
        description=(
            'If set, will try to discover a trigger for this LoRA which matches or is similar to this string '
            'and inject it into the prompt. '
            'If set to "any", it will be pick the first trigger.'
        ),
    )
    is_version: bool | None = Field(
        default=None,
        description=(
            "If true, the LoRA ID will be treated as a CivitAI model version ID. "
            "This requires the name to be a valid integer."
        ),
    )


class TextualInversion(HordeModel):
    identifier: str = RenamedField(
        description="The exact name or CivitAI ID of the Textual Inversion.",
        renamed_to="identifier", original_name="name",
    )
    injection_location: TIPlacement | None = RenamedField(
        default=None,
        description=(
            "If set, will automatically inject this TI (filename and strength) into the specified prompt. "
            "If unset, the user will have to manually add the TI filename to the desired prompt."
        ),
        renamed_to="injection_location", original_name="inject_ti",
    )
    strength: float | None = Field(
        default=None,
        description="The strength with which to apply the TI to the prompt. Only used when inject_ti set.",
        ge=-5, le=5,
    )

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def fix_ids_having_two_names(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "id" in values:
            # Why?
            # TODO: Also renamed fields are wacky and I cant validate my renamed field???
            #  What was past me *thinking*?
            values["name"] = str(values.pop("id"))
        return values


class ExtraSourceImage(HordeModel):
    image: Base64Image | str = Field(
        description="The source image.",
    )
    strength: float | None = Field(
        default=None,
        description="The strength with which to apply the image to the image generation model.",
        ge=-5, le=5,
    )


class ImageGenerationParams(HordeModel):
    width: int | None = Field(
        default=None,
        description="The width of the image to generate",
        ge=64, le=3072, multiple_of=64,
    )
    height: int | None = Field(
        default=None,
        description="The height of the image to generate",
        ge=64, le=3072, multiple_of=64,
    )

    image_count: int | None = RenamedField(
        default=None,
        description="The number of images to generate.",
        renamed_to="image_count", original_name="n",
        ge=1, le=20,
    )

    seed: str | None = Field(
        default=None,
        description="The seed to use to generate this request. Supports text and numbers.",
    )
    seed_variation: int | None = Field(
        default=None,
        description="If multiple images are requested, each image's seed will be the previous seed plus this value.",
        ge=1, le=1000,
    )

    sampler: Sampler | str | None = RenamedField(
        default=None,
        description="The sampler to use when generating this request.",
        renamed_to="sampler", original_name="sampler_name",
    )
    steps: int | None = Field(
        default=None,
        description="The number of sampling steps to perform when generating this request. ",
        ge=1, le=500,
    )
    denoising_strength: float | None = Field(
        default=None,
        description=(
            "The denoising strength determines how much noise is initially added to the starting image. "
            "Higher values will result in the output image differing more from the starting image "
            "(given a high enough step count)."
        ),
        ge=0.01, le=1.0,
    )
    cfg_scale: float | None = Field(
        default=None,
        description=(
            "The CFG scale (classifier-free guidance scale) to use when generating the request. "
            "Higher values makes the image follow the prompt more closely."
        ),
        ge=0.0, le=100.0,
    )
    post_processors: UniqueList[PostProcessor] | None = RenamedField(
        default=None,
        description="A list of post-processors to apply to the image, in the order specified.",
        renamed_to="post_processors", original_name="post_processing",
    )
    facefixer_strength: float | None = Field(
        default=None,
        description="How much to attempt to fix faces in the image. Requires a face-fixing post-processor.",
        ge=0.0, le=1.0,
    )

    loras: conlist(LoRA, max_length=5) | None = Field(
        default=None,
        description="A list of LoRAs to use when generating this request.",
    )
    textual_inversions: list[TextualInversion] | None = RenamedField(
        default=None,
        description="A list of Textual Inversions to use when generating this request.",
        renamed_to="textual_inversions", original_name="tis",
    )

    control_type: ControlType | None = Field(
        default=None,
        description="The type of ControlNet to use when generating this request.",
    )
    image_is_control: bool | None = Field(
        default=None,
        description=(
            "If true, the source image is a pre-generated control map for ControlNet use, "
            "otherwise a control map will be generated."
        ),
    )
    return_control_map: bool | None = Field(
        default=None,
        description="If true, the generated control map will be returned instead of the image.",
    )

    clip_skip: int | None = Field(
        default=None,
        description="The number of CLIP language processor layers to skip.",
        ge=1, le=12,
    )
    hires_fix: bool | None = Field(
        default=None,
        description="If true, process the image at base resolution before upscaling and re-processing.",
    )
    karras: bool | None = Field(
        default=None,
        description="If true, enable karras noise scheduling tweaks.",
    )
    tiling: bool | None = Field(
        default=None,
        description="If true, create images that stitch together seamlessly.",
    )
    special: dict | None = Field(
        default=None,
        description="Reserved field for special cases, should generally not be used.",
    )


class ImageGenerationRequest(HordeRequest):
    @computed_field
    @property
    def prompt(self) -> str:
        prompt = self.positive_prompt
        if self.negative_prompt is not None:
            prompt += f"###{self.negative_prompt}"

        # Prevents the replacement filter from being ignored due to prompt length
        if self.replacement_filter and len(prompt) >= 1000:
            prompt = prompt[:1000]

        return prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        values = value.split("###", 1)
        self.positive_prompt = values[0]
        if len(values) > 1:
            self.negative_prompt = values[1]

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def prompt_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "positive_prompt" not in values and "prompt" in values:
            prompt_parts = values.pop("prompt").split("###", 1)
            values["positive_prompt"] = prompt_parts[0]
            if len(prompt_parts) > 1:
                values["negative_prompt"] = prompt_parts[1]
        return values

    positive_prompt: str = Field(
        min_length=1,
        description="The positive prompt which will be used to generate the image(s).",
        exclude=True,  # Not in the API, see self.prompt
    )
    negative_prompt: str | None = Field(
        default=None,
        description="The negative prompt which will be used to generate the image(s).",
        exclude=True,  # Not in the API, see self.prompt
    )

    params: ImageGenerationParams | None = None

    nsfw: bool | None = Field(
        default=None,
        description="Set to true if this request is NSFW. This will skip workers which censor images.",
    )
    censor_nsfw: bool | None = Field(
        default=None,
        description="If true, NSFW images accidentally generated by a request not marked as NSFW will be censored.",
    )
    replacement_filter: bool | None = Field(
        default=None,
        description=(
            "If true, prompts found to be suspicious (CSAM) "
            "will have their suspicious parts replaced with safer alternatives. "
            "If false, the request will be blocked and the user's IP will get a timeout. "
            # "This setting is ignored and treated as false if the prompt exceeds 1000 characters."
            "For further information, see "
            "https://github.com/Haidra-Org/AI-Horde/blob/main/FAQ.md#can-you-explain-how-the-anti-csam-regex-filter-works"
        ),
    )

    models: list[str] | None = Field(
        default=None,
        description=(
            "A list of models that are allowed to fulfill this request. "
            "If none are specified, all models are allowed."
        ),
    )

    source_image: Base64Image | str | None = Field(
        default=None,
        description="A base64 encoded WEBP image to use for img2img.",
    )
    source_processing: SourceProcessing | None = Field(
        default=None,
        description=(
            "If a source image is provided, this specifies how it should be processed. "

            "If set to inpainting or outpainting, a mask must also be provided "
            "either via source_mask or as the source image's alpha channel. "
        ),
    )
    source_mask: str | None = Field(
        default=None,
        description=(
            "A base64 encoded WEBP image to use for inpainting and outpainting of the source image. "
            "If not provided, the source image's alpha channel will be used as the mask. "
            # TODO: Factcheck this
            # "The mask should be a grayscale image where white represents the area to inpaint/outpaint "
            # "and black represents the area to keep. "
        ),
    )
    extra_source_images: conlist(ExtraSourceImage, max_length=5) | None = Field(
        default=None,
        description=(
            "A list of extra images to use for stable cascade remixing. "
            'source_processing must be set to "remix"'
        ),  # TODO: Expand upon explanation
    )

    workers: conlist(str, max_length=5) | None = Field(
        default=None,
        description="Up to 5 workers which are allowed to service this request.",
    )
    worker_blacklist: bool | None = Field(
        default=None,
        description="If true, the workers list will be treated as a blacklist instead of a whitelist.",
    )
    trusted_workers: bool | None = Field(
        default=None,
        description="If true, only trusted workers will be allowed to service this request.",
    )
    slow_workers: bool | None = Field(
        default=None,
        description=(
            "If true, slower workers will be allowed to service this request. "
            "Disabling this increases the kudo cost of the request."
        ),
    )

    r2: bool | None = Field(
        default=None,
        description="If True, the image will be returned via cloudflare r2.",
    )

    dry_run: bool | None = Field(
        default=None,
        description="If true, the estimated kudo cost of the request will be returned instead of generating an image.",
    )

    allow_downgrade: bool | None = Field(
        default=None,
        description=(
            "If true, the request will be allowed to be downgraded in case the upfront kudos cost is not met. "
            "This means that resolution will be lowered while keeping the aspect ratio the same."
        ),
    )

    shared: bool | None = Field(
        default=None,
        description=(
            "If true, the generated image will be shared with LAION (https://laion.ai/) for improving their dataset. "
            "This will reduce the kudo cost of the request by 2. "
            "If the user is anonymous, this will allways be treated as true."
        ),
    )

    proxied_account: str | None = Field(
        default=None,
        description=(
            "If using a service account as a proxy, "
            "provide this value to identify the actual account from which this request is coming from."
        ),
    )

    def apply_style(self, style: Style, cache: Cache) -> ImageGenerationRequest:
        return style.to_generation_request(self, cache=cache)
# endregion


# region Generation status
class GenerationMetadataType(StrEnum):
    CENSORSHIP = "censorship"
    SOURCE_IMAGE = "source_image"
    SOURCE_MASK = "source_mask"
    BATCH_INDEX = "batch_index"
    LORA = "lora"
    TI = "ti"


# noinspection SpellCheckingInspection
class GenerationMetadataValue(StrEnum):
    DOWNLOAD_FAILED = "download_failed"
    PARSE_FAILED = "parse_failed"
    BASELINE_MISMATCH = "baseline_mismatch"
    CSAM = "csam"
    NSFW = "nsfw"
    SEE_REF = "see_ref"


class GeneratedImageMetadata(HordeModel):
    type: GenerationMetadataType = Field(
        description="The relevance of the metadata field.",
    )
    cause: GenerationMetadataValue | str | None = Field(
        default=None,
        description="The value of the metadata field.",
    )
    ref: str | None = Field(
        default=None,
        description="A reference for the metadata (e.g. a lora ID).",
        max_length=255,
    )


class FinishedImageGeneration(HordeModel):
    img: Base64Image | str = Field(
        description=(
            "The generated image. "
            "This can be either the image data as a Base64-encoded .webp file or a URL to the image, "
            "depending on the `r2` parameter of the request."
        ),
    )

    # noinspection PyNestedDecorators
    @field_validator("img", mode="plain")
    @classmethod
    def validate_img(cls, value: str) -> Base64Image | str:
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return Base64Image(value)

    id: str = Field(
        description="The ID of the generated image.",
    )
    worker_id: str = Field(
        description="The UUID of the worker which generated this image.",
    )
    worker_name: str = Field(
        description="The name of the worker which generated this image.",
    )
    model: str = Field(
        description="The name of the model which generated this image.",
    )
    seed: str = Field(
        description="The seed used to generate this image.",
    )

    censored: bool = Field(
        description="When true this image has been censored by the worker's safety filter.",
    )
    gen_metadata: list[GeneratedImageMetadata] | None = Field(
        description="A list of metadata fields for this image.",
    )
    # TODO: When https://github.com/pydantic/pydantic/issues/2255 gets implemented, mark this as deprecated properly
    state: Literal["ok", "censored"] = Field(
        description="DEPRECATED! The state of this generation. Made obsolete by the `gen_metadata` field.",
    )


class ImageGenerationStatus(GenerationCheck):
    generations: list[FinishedImageGeneration] = Field(
        description="A list of generated images.",
    )
    shared: bool = Field(
        description="True if the image was shared with LAION (https://laion.ai/) for improving their dataset.",
    )
# endregion


# region Interrogation
# noinspection SpellCheckingInspection
class InterrogationTypeImage(StrEnum):
    GFPGAN = "GFPGAN"
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    NMKD_SIAX = "NMKD_Siax"
    FOURX_ANIMESHARP = "4x_AnimeSharp"
    CODEFORMERS = "CodeFormers"
    STRIP_BACKGROUND = "strip_background"


# noinspection SpellCheckingInspection
class InterrogationType(StrEnum):
    CAPTION = "caption"
    INTERROGATION = "interrogation"
    NSFW = "nsfw"

    # FUn fact, you can't extend an enum, so we need to duplicate the values here!
    GFPGAN = InterrogationTypeImage.GFPGAN
    REALESRGAN_X4PLUS = InterrogationTypeImage.REALESRGAN_X4PLUS
    REALESRGAN_X2PLUS = InterrogationTypeImage.REALESRGAN_X2PLUS
    REALESRGAN_X4PLUS_ANIME_6B = InterrogationTypeImage.REALESRGAN_X4PLUS_ANIME_6B
    NMKD_SIAX = InterrogationTypeImage.NMKD_SIAX
    FOURX_ANIMESHARP = InterrogationTypeImage.FOURX_ANIMESHARP
    CODEFORMERS = InterrogationTypeImage.CODEFORMERS
    STRIP_BACKGROUND = InterrogationTypeImage.STRIP_BACKGROUND


class InterrogationRequestForm(HordeModel):
    name: InterrogationType = Field(
        description="The type of interrogation this is.",
    )
    # TODO: Figure out what this is
    payload: dict | None = Field(
        default=None,
    )


class InterrogationRequest(HordeRequest):
    image_url: str = RenamedField(
        description="URL of the image to interrogate.",
        renamed_to="image_url", original_name="source_image",
    )
    forms: list[InterrogationRequestForm] = Field(
        description="A list of forms to use when interrogating the image.",
    )
    slow_workers: bool | None = Field(
        default=None,
        description=(
            "If true, slower workers will be allowed to service this request. "
            "Disabling this increases the kudo cost of the request."
        ),
    )


class InterrogationResponse(HordeSuccess):
    id: str = Field(
        description="The UUID of the request. Use this to retrieve the request status in the future.",
    )
    message: str | None = Field(
        default=None,
        description="Any extra information from the horde about this request.",
    )


class InterrogationStatusState(StrEnum):
    WAITING = "waiting"
    PROCESSING = "processing"
    DONE = "done"
    FAULTED = "faulted"
    PARTIAL = "partial"


class CaptionResult(HordeModel):
    caption: str = Field(
        description="The caption generated for the image.",
    )


class InterrogationCategoryItem(HordeModel):
    """Represents a describing term and its confidence."""

    text: str = Field(
        description="The actual value.",
    )
    confidence: float = Field(
        description="The confidence of this value being applicable to the image.",
    )


class InterrogationCategories(HordeModel):
    tags: list[InterrogationCategoryItem]
    sites: list[InterrogationCategoryItem]
    artists: list[InterrogationCategoryItem]
    flavors: list[InterrogationCategoryItem]
    mediums: list[InterrogationCategoryItem]
    movements: list[InterrogationCategoryItem]
    techniques: list[InterrogationCategoryItem]


class InterrogationResult(HordeModel):
    interrogation: InterrogationCategories


class NSFWResult(HordeModel):
    nsfw: bool = Field(
        description="Weather the image has been classified as NSFW.",
    )


class GenericProcessedImageResult(HordeModel):
    def __init__(self, **data: str) -> None:
        image_url = None
        for key in InterrogationTypeImage:
            if key in data:
                image_url = data.pop(key)
                break
        if image_url:
            data["image_url"] = image_url
        super().__init__(**data)

    image_url: str = Field(
        description="The URL of the resulting image.",
    )


class InterrogationStatusForm(HordeModel):
    form: InterrogationType = Field(
        description="The name of this interrogation form.",
    )
    state: InterrogationStatusState = Field(
        description="The overall status of this interrogation.",
    )
    result: CaptionResult | InterrogationResult | NSFWResult | GenericProcessedImageResult | None = Field(
        default=None,
        description="The result of this interrogation form.",
    )


class InterrogationStatus(HordeSuccess):
    state: InterrogationStatusState = Field(
        description="The overall status of this interrogation.",
    )
    forms: list[InterrogationStatusForm] = Field(
        description="A list of forms with their results.",
    )
# endregion
