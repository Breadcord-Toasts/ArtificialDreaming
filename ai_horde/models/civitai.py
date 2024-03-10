import datetime
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .general import HordeModel, HordeSuccess, RenamedField


# noinspection SpellCheckingInspection
class ModelType(StrEnum):
    CHECKPOINT = "Checkpoint"
    TEXTUALINVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETICGRADIENT = "AestheticGradient"
    LORA = "LORA"
    LOCON = "LoCon"
    CONTROLNET = "Controlnet"
    POSES = "Poses"
    WORKFLOWS = "Workflows"
    OTHER = "Other"


class ScanResult(StrEnum):
    PENDING = "Pending"
    SUCCESS = "Success"
    DANGER = "Danger"
    ERROR = "Error"


# noinspection SpellCheckingInspection
class CivitAIFileFormat(StrEnum):
    SAFETENSOR = "SafeTensor"
    PICKLETENSOR = "PickleTensor"
    OTHER = "Other"
    DIFFUSERS = "Diffusers"


class NSFWLevel(StrEnum):
    MATURE = "Mature"
    SOFT = "Soft"
    X = "X"


def validate_dimensions(value: str | Sequence[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    # case of already parsed data being passed
    if isinstance(value, Sequence) and not isinstance(value, str):
        assert len(value) == 2, "Must be a tuple of length 2"
        return value[0], value[1]  # Make the type checker happy, so it gets tuple[int, int] and not tuple[int, ...]

    assert isinstance(value, str), "Must be a string"
    split = value.split("x", maxsplit=1)
    assert len(split) == 2, 'Must be in the format "WIDTHxHEIGHT"'
    return int(split[0]), int(split[1])


Dimensions = Annotated[tuple[int, int], BeforeValidator(validate_dimensions)]


class CivitAIImageMeta(HordeModel):
    """Generation parameters for a CivitAI image. Further fields will likely be added at runtime."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    # Fields are so dynamic that it's not a nice UX to raise an exception if one doesn't exist
    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            return None

    prompt: str | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    size: Dimensions | None = Field(default=None)
    hires_resize: Dimensions | None = Field(default=None)
    seed: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    sampler: str | None = Field(default=None)
    cfg_scale: float | None = Field(default=None)
    clip_skip: int | None = Field(default=None)
    hires_steps: int | None = Field(default=None)
    denoising_strength: float | None = Field(default=None)

    model: str | None = Field(default=None)
    vae: str | None = Field(default=None)
    hashes: dict[str, str] | None = Field(default=None)

    resources: list[dict[str, Any]] | None = Field(default=None)

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def fix_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        def fix_stupid_key_name(key: str) -> str:
            key = key.replace(" ", "_")
            snake_case_key = ""
            for index, char in enumerate(key):
                if char.isupper() and index != 0 and len(key) > index + 1 and not key[index + 1].isupper():
                    snake_case_key += "_"
                snake_case_key += char.lower()
            return snake_case_key

        return {fix_stupid_key_name(key): value for key, value in data.items()}


class CivitAIImageMetadata(HordeModel):
    model_config = ConfigDict(extra="allow")

    hash: str = Field(
        description="The image's hash.",
    )
    size: int | None = Field(
        default=None,
        description="The image's size.",
    )
    width: int = Field(
        description="The image's width.",
    )
    height: int = Field(
        description="The image's height.",
    )
    seed: int | None = Field(
        default=None,
        description="The image's seed.",
    )


class CivitAIImage(HordeSuccess):
    model_config = ConfigDict(extra="allow")

    id: int | None = Field(
        default=None,
        description="The image's identifier.",
    )
    url: str = Field(
        description="The image's URL.",
    )
    width: int = Field(
        description="The image's width.",
    )
    height: int = Field(
        description="The image's height.",
    )
    nsfw: NSFWLevel | None = Field(
        default=None,
        description="The image's NSFW level.",
    )
    hash: str = Field(
        description="The image's hash.",
    )
    metadata: CivitAIImageMetadata = Field(
        description="The image's metadata, such as its hash and size.",
    )
    meta: CivitAIImageMeta | None = Field(
        default=None,
        description="Meta info about the image, like generation parameters.",
    )
    type: Literal["image", "video"] | None

    # noinspection PyNestedDecorators
    @field_validator("nsfw", mode="before")
    @classmethod
    def validate_nsfw(cls, value: str) -> NSFWLevel | None:
        if value == "None" or value is None:
            return None
        return NSFWLevel(value)


class CivitAICreator(HordeSuccess):
    # Removes a warning about field names starting with "model_" being reserved
    model_config = ConfigDict(protected_namespaces=())

    username: str = Field(
        description="The creator's username. This acts as a unique identifier.",
    )
    image_url: str | None = RenamedField(
        default=None,
        description="The creator's image (avatar).",
        renamed_to="image_url", original_name="image",
    )
    model_count: int | None = RenamedField(
        default=None,
        description="The number of models the creator has uploaded.",
        renamed_to="model_count", original_name="modelCount",
    )
    uploaded_models_url: str | None = RenamedField(
        default=None,
        description="Link to the creator's uploaded models.",
        renamed_to="uploaded_models_url", original_name="link",
    )


class CivitAIModelStats(HordeModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    download_count: int = RenamedField(
        description="The number of times the model has been downloaded.",
        renamed_to="download_count", original_name="downloadCount",
    )
    favorite_count: int | None = RenamedField(
        default=None,
        description="The number of times the model has been favorited.",
        renamed_to="favorite_count", original_name="favoriteCount",
    )
    comment_count: int | None = RenamedField(
        default=None,
        description="The number of comments the model has.",
        renamed_to="comment_count", original_name="commentCount",
    )
    tipped_amount_count: int | None = RenamedField(
        default=None,
        description="The number of times the model has been tipped.",
        renamed_to="tipped_amount_count", original_name="tippedAmountCount",
    )
    ratingCount: int = RenamedField(
        description="The number of ratings the model has.",
        renamed_to="ratingCount", original_name="ratingCount",
    )
    rating: float | None = Field(
        description="The model's rating.",
    )


class CivitAIModelFileMetadata(HordeModel):
    model_config = ConfigDict(extra="allow")

    fp: str | None = Field(
        default=None,
        description="The model file's floating point precision, such as fp16 or fp32",
    )
    size: Literal["full", "pruned"] | None = Field(
        default=None,
        description="The model's size (full or pruned).",
    )
    format: CivitAIFileFormat | None = Field(
        default=None,
        description="The model file's format.",
    )
    hash: str | None = RenamedField(
        default=None,
        description="The model file's hash.",
        renamed_to="hash", original_name="modelFileHash",
    )
    id: int | None = RenamedField(
        default=None,
        description="The model file's identifier.",
        renamed_to="id", original_name="modelFileId",
    )
    vae_migration: bool | None = RenamedField(
        default=None,
        renamed_to="vae_migration", original_name="vaeMigration",
    )


class CivitAIModelFile(HordeModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    name: str = Field(
        description="The file's name.",
    )
    id: int = Field(
        description="The file's identifier.",
    )
    # TODO: How is this different to the ID?
    model_version_id: int | None = RenamedField(
        default=None,
        description="The model version's identifier.",
        renamed_to="model_version_id", original_name="modelVersionId",
    )
    download_url: str = RenamedField(
        description="The file's download URL.",
        renamed_to="download_url", original_name="downloadUrl",
    )
    hashes: dict[str, str] = Field(
        description="The file's hashes.",
    )
    size_kb: float | None = RenamedField(
        default=None,
        description="The file's size in kilobytes.",
        renamed_to="size_kb", original_name="sizeKB",
    )
    pickle_scan: ScanResult = RenamedField(
        description="The status of the pickle scan.",
        renamed_to="pickle_scan", original_name="pickleScanResult",
    )
    pickle_scan_message: str | None = RenamedField(
        default=None,
        description="The message from the pickle scan.",
        renamed_to="pickle_scan_message", original_name="pickleScanMessage",
    )
    virus_scan: ScanResult = RenamedField(
        description="The status of the virus scan.",
        renamed_to="virus_scan", original_name="virusScanResult",
    )
    virus_scan_message: str | None = RenamedField(
        default=None,
        description="The message from the virus scan.",
        renamed_to="virus_scan_message", original_name="virusScanMessage",
    )
    scanned_at: datetime.datetime | None = RenamedField(
        default=None,
        description="The date the file was scanned.",
        renamed_to="scanned_at", original_name="scannedAt",
    )
    metadata: CivitAIModelFileMetadata = Field(
        description="The file's metadata.",
    )
    primary: bool | None = Field(
        default=None,
        description="Whether the file is the primary file.",
    )
    type: Literal["Model", "Pruned Model", "VAE", "Config", "Training Data", "Archive", "Negative"] | str


class BaseModelType(StrEnum):
    STANDARD = "Standard"
    INPAINTING = "Inpainting"


class CivitAIModelVersion(HordeModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    id: int = Field(
        description="The model version's identifier.",
    )
    model_id: int = RenamedField(
        description="The model version's model identifier.",
        renamed_to="model_id", original_name="modelId",
    )
    name: str = Field(
        description="The model version's name.",
    )
    description: str | None = Field(
        default=None,
        description="The model version's description as HTML. Usually a changelog.",
    )
    created_at: datetime.datetime = RenamedField(
        description="The model version's creation date.",
        renamed_to="created_at", original_name="createdAt",
    )
    updated_at: datetime.datetime = RenamedField(
        description="The model version's last update date.",
        renamed_to="updated_at", original_name="updatedAt",
    )
    published_at: datetime.datetime | None = RenamedField(
        default=None,
        description="The model version's publication date.",
        renamed_to="published_at", original_name="publishedAt",
    )
    download_url: str = RenamedField(
        description="The model version's download URL.",
        renamed_to="download_url", original_name="downloadUrl",
    )
    trained_words: list[str] = RenamedField(
        description="The words used to trigger the model.",
        renamed_to="trained_words", original_name="trainedWords",
    )
    files: list[CivitAIModelFile]
    images: list[CivitAIImage]
    # TODO: Are there any other states this can be in? There shouldn't be, right?
    #  Will be hard to test since only published models should be on the homepage, so might not be a real issue
    status: Literal["Published"] = Field(
        description="The model version's publication status.",
    )
    stats: CivitAIModelStats
    vae_id: int | None = RenamedField(
        default=None,
        renamed_to="vae_id", original_name="vaeId",
    )
    early_axes_time_frame: int | None = RenamedField(
        default=None,
        renamed_to="early_axes_time_frame", original_name="earlyAccessTimeFrame",
    )
    base_model: str = RenamedField(
        renamed_to="base_model", original_name="baseModel",
    )
    base_model_type: BaseModelType | None = RenamedField(
        default=None,
        renamed_to="base_model_type", original_name="baseModelType",
    )

    @property
    def thumbnail_url(self) -> str:
        return self.images[0].url

    @property
    def sfw_thumbnail_url(self) -> str | None:
        for image in self.images:
            if image.nsfw in (NSFWLevel.SOFT, None):
                return image.url
        return None

    @property
    def url(self) -> str:
        return f"https://civitai.com/models/{self.model_id}?modelVersionId={self.id}"


class CivitAIModel(HordeSuccess):
    id: int = Field(
        description="THe model identifier.",
    )
    name: str = Field(
        description="The model name.",
    )
    description: str | None = Field(
        default=None,
        description="The model description as HTML.",
    )
    type: ModelType = Field(
        description="The model type.",
    )
    nsfw: bool = Field(
        description="Whether the model is NSFW or not.",
    )
    tags: list[str] = Field(
        description="The model's associated tags.",
    )
    state: Literal["Archived", "TakenDown"] | None = RenamedField(
        default=None,
        description=(
            "The model's current mode. "
            "If it's archived, the files field will be empty. "
            "If it's taken down, the images field will be empty."
        ),
        renamed_to="state", original_name="mode",
    )
    creator: CivitAICreator = Field(
        description="The model's creator.",
    )
    versions: list[CivitAIModelVersion] = RenamedField(
        description="The model's versions.",
        renamed_to="versions", original_name="modelVersions",
    )
    poi: bool = Field(
        description="Whether the model is a point of interest.",
    )
    stats: CivitAIModelStats = Field(
        description="The model's stats.",
    )

    allow_different_license: bool = RenamedField(
        # TODO: write description
        renamed_to="allow_different_license", original_name="allowDifferentLicense",
    )
    allow_derivatives: bool = RenamedField(
        description="Whether derivative models are allowed to be created.",
        renamed_to="allow_derivatives", original_name="allowDerivatives",
    )
    allow_no_credit: bool = RenamedField(
        # TODO: write description
        renamed_to="allow_no_credit", original_name="allowNoCredit",
    )
    # Sorry, but you're on your own on this one. I can not be bothered
    allow_commercial_use: list[Any] | None = RenamedField(
        default=None,
        renamed_to="allow_commercial_use", original_name="allowCommercialUse",
    )

    @property
    def thumbnail_url(self) -> str:
        return self.versions[0].thumbnail_url

    @property
    def sfw_thumbnail_url(self) -> str | None:
        for version in self.versions:
            if url := version.sfw_thumbnail_url:
                return url
        return None

    @property
    def url(self) -> str:
        return f"https://civitai.com/models/{self.id}"


class SearchCategory(StrEnum):
    MODELS = "models_v5"
    IMAGES = "images_v3"
    USERS = "users_v2"
    ARTICLES = "articles_v3"
    BOUNTIES = "bounties"
    COLLECTIONS = "collections"


class SearchFilter:
    def __init__(self) -> None:
        self._filters: dict[str, list[str]] = defaultdict(list)

    @property
    def serialize(self) -> list[list[str]]:
        return [
            [f'"{key}"="{value}"' for value in values]
            for key, values in self._filters.items()
        ]

    def model_type(self, model_type: ModelType, /) -> "SearchFilter":
        self._filters["type"].append(model_type)
        return self

    def base_model_type(self, base_model: str, /) -> "SearchFilter":
        self._filters["version.baseModel"].append(base_model)
        return self
