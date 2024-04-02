from collections.abc import Mapping, MutableMapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .general import HordeModel, RenamedField
from .image import ImageGenerationParams, ImageGenerationRequest, LoRA, Sampler, TextualInversion

if TYPE_CHECKING:
    from ..cache import Cache


# region Styles

class StyleBase(HordeModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(
        default=None,
        description="The model to use for the image generation.",
    )
    steps: int | None = Field(
        default=None,
        description="The number of sampling steps to perform when generating this request. ",
        ge=1, le=500,
    )
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
    cfg_scale: float | None = Field(
        default=None,
        description=(
            "The CFG scale (classifier-free guidance scale) to use when generating the request. "
            "Higher values makes the image follow the prompt more closely."
        ),
        ge=0.0, le=100.0,
    )
    sampler: Sampler | None = RenamedField(
        default=None,
        description="The sampler to use when generating this request.",
        renamed_to="sampler", original_name="sampler_name",
    )
    loras: list[LoRA] | None = Field(
        default=None,
        description="A list of LoRAs to use when generating this request.",
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
    textual_inversions: list[TextualInversion] | None = RenamedField(
        default=None,
        description="A list of Textual Inversions to use when generating this request.",
        renamed_to="textual_inversions", original_name="tis",
    )


class StyleEnhancement(StyleBase):
    pass


class Style(StyleBase):
    name: str = Field(
        min_length=1,
        description="The name of the style.",
    )
    enhance: bool = Field(
        default=False,
        description="If true, add enhancements to the style according ot its base model.",
    )

    positive_prompt: str = Field(
        min_length=1,
        description="The positive prompt which will be used to generate the image(s).",
        pattern=r".*\{positive_prompt\}.*",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="The negative prompt which will be used to generate the image(s).",
        pattern=r".*\{negative_prompt\}.*",
    )

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def fix_prompt(cls, data: Any) -> Any:
        assert isinstance(data, dict), "Data must be a dict"
        if "prompt" not in data:
            return data

        prompt = data.pop("prompt")
        assert isinstance(prompt, str), "Prompt must be a string"
        prompt = prompt.replace("{p}", "{positive_prompt}")
        prompt = prompt.replace("{np}", "{negative_prompt}")

        if "###" not in prompt and "{negative_prompt}" in prompt:
            prompt = prompt.replace("{negative_prompt}", "###{negative_prompt}")

        prompts = last_split(prompt, "###")
        data["positive_prompt"] = prompts[0]
        data["negative_prompt"] = prompts[1] if len(prompts) > 1 else None

        return data

    @property
    def prompt(self) -> str | None:
        prompt = self.positive_prompt
        if prompt is not None and self.negative_prompt is not None:
            prompt += f"###{self.negative_prompt}"
        return prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        values = value.split("###", 1)
        self.positive_prompt = values[0]
        if len(values) > 1:
            self.negative_prompt = values[1]

    def to_generation_request(
            self,
            base_request: ImageGenerationRequest,
            cache: "Cache",
    ) -> ImageGenerationRequest:
        style = self.model_dump(by_alias=False)
        style.pop("name")

        positive_prompt = style.pop("positive_prompt").replace("{positive_prompt}", base_request.positive_prompt)
        negative_prompt: str | None = style.pop("negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = base_request.negative_prompt
        elif base_request.negative_prompt is not None:
            negative_prompt = negative_prompt.replace("{negative_prompt}", base_request.negative_prompt)

        if style.pop("enhance", False):
            model = style["model"]
            baseline = cache.horde_model_reference[model].baseline
            enhancements = cache.enhancements[baseline]
            recursive_update(style, enhancements.model_dump(by_alias=False, exclude_none=True), overwrite_none=True)

        dummy_params = {
            "loras": (
                [LoRA.model_validate(lora, strict=True) for lora in style.pop("loras") or []]
                + ((base_request.params or ImageGenerationParams()).loras or [])
            ) or None,
            "textual_inversions": (
                [TextualInversion.model_validate(ti, strict=True) for ti in style.pop("textual_inversions") or []]
                + ((base_request.params or ImageGenerationParams()).textual_inversions or [])
            ) or None,
        }
        dummy_request = base_request.model_copy(
            deep=True,
            update={
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "models": [style.pop("model")] if style.get("model") else base_request.models,
                "params": base_request.params.model_copy(
                    deep=True,
                    update=dummy_params,
                ) if base_request.params else ImageGenerationParams(**dummy_params),
            },
        )

        # noinspection Pydantic
        request_fields = set(ImageGenerationRequest.model_fields.keys())
        # noinspection Pydantic
        params_fields = set(ImageGenerationParams.model_fields.keys())
        for field, value in style.items():
            if value is None:
                continue

            if field in request_fields:
                setattr(dummy_request, field, value)
            elif field in params_fields:
                setattr(dummy_request.params, field, value)
            else:
                print("Unknown field:", field, value)

        return ImageGenerationRequest.model_validate(dummy_request, strict=True)
# endregion


# region Model Reference
class SupportedFeature(StrEnum):
    HIRES_FIX = "hires_fix"
    LORAS = "loras"
    INPAINTING = "inpainting"
    CONTROLNET = "controlnet"


class ModelReferenceFile(HordeModel):
    path: str = Field(
        description="Model file filename.",
    )
    sha256sum: str | None = Field(
        default=None,
        description="SHA256 hash of the file.",
    )
    md5sum: str | None = Field(
        default=None,
        description="MD5 hash of the file.",
    )


class ModelReferenceDownload(HordeModel):
    file_name: str
    file_path: str
    file_url: str


class ModelReferenceConfig(HordeModel):
    files: list[ModelReferenceFile]
    download: list[ModelReferenceDownload]


class ModelReference(HordeModel):
    name: str = Field(
        description="The model name, as it's known by the horde.",
    )
    description: str = Field(
        description="A description of the model.",
    )
    baseline: str = Field(
        description="Name of the model this one is based on.",
    )
    homepage: str | None = None
    version: str
    nsfw: bool
    style: Literal["generalist", "anime", "realistic", "artistic", "furry", "other"] | str = Field(
        description="The style of the model's generations.",
    )
    type: Literal["ckpt"] = Field(
        description="The model file type.",
    )
    showcases: list[str] = Field(
        default=[],
        description="A list of image URLs showcasing outputs form the model.",
    )
    config: ModelReferenceConfig
    available: bool | None = None
    size_on_disk_bytes: int | None = None
    tags: list[str] = Field(
        default=[],
        description="A list of tags describing the model.",
    )
    # TODO: Rename to "triggers"
    trigger: list[str] = Field(
        default=[],
        description="A list of model trigger words.",
    )
    # TODO: Rename to "unsupported_features"
    features_not_supported: list[SupportedFeature] = Field(
        default=[],
        description="A list unsupported features.",
    )
    download_all: bool
    inpainting: bool
    min_bridge_version: int | None = None
# endregion


def last_split(string: str, sep: str) -> list[str]:
    """Split a string by a separator, and return the text after the last instance of the separator, and the rest."""
    split = string.split(sep)
    if len(split) <= 2:
        return split

    return [sep.join(split[:-1]), split[-1]]


def recursive_update(target: MutableMapping[str, Any], source: Mapping[str, Any], overwrite_none: bool = False) -> None:
    if not isinstance(target, MutableMapping):
        raise ValueError(f"Target is not a MutableMapping: {target}")
    for key, value in source.items():
        if overwrite_none and target.get(key) is None:
            target[key] = value
            continue
        if isinstance(value, MutableMapping):
            recursive_update(target.setdefault(key, {}), value, overwrite_none=overwrite_none)
        elif isinstance(value, Sequence):
            target.setdefault(key, []).extend(value)
        else:
            target[key] = value
