from typing import Literal

from pydantic import Field, conlist

from .general import HordeModel, HordeRequest, RenamedField
from .horde_meta import GenerationCheck


# noinspection SpellCheckingInspection
class TextGenerationParams(HordeModel):
    # TODO: Document fields with missing descriptions

    count: int | None = RenamedField(
        None,
        description="Number of texts to generate.",
        renamed_to="count", original_name="n",
    )
    use_default_bad_words: bool | None = RenamedField(
        None,
        description="If true, use the default KoboldAI bad word IDs.",
        renamed_to="use_default_bad_words", original_name="use_default_badwordsids",
    )
    stop_sequences: list[str] | None = RenamedField(
        None,
        description=(
            "A list of string sequences, which will stop the generation of further tokens. "
            "The returned text will contain the stop sequence."
        ),
        renamed_to="stop_sequences", original_name="stop_sequence",
    )
    max_length: int | None = Field(
        None,
        description="Maximum number of tokens to generate.",
        ge=16, le=512,
    )
    max_context_length: int | None = Field(
        None,
        description="Maximum number of tokens to send to the model.",
        ge=80, le=32000,
    )

    repetition_penalty: float | None = RenamedField(
        None,
        description="Controls how much repetition is allowed in the generated text.",
        renamed_to="repetition_penalty", original_name="rep_pen",
        ge=1.0, le=3.0,
    )
    repetition_penalty_range: int | None = RenamedField(
        None,
        renamed_to="repetition_penalty_range", original_name="rep_pen_range",
        ge=1, le=4096,
    )
    repetition_penalty_slope: float | None = RenamedField(
        None,
        description="Controls how much repetition is allowed in the generated text.",
        renamed_to="repetition_penalty_slope", original_name="rep_pen_slope",
        ge=1.0, le=3.0,
    )

    temperature: float | None = Field(
        None,
        description='Controls the randomness, or "creativity" of the generated text.',
        ge=0.0, le=5.0,
    )
    dynamic_temp_range: float | None = RenamedField(
        None,
        renamed_to="dynamic_temp_range", original_name="dynatemp_range",
        ge=0.0, le=5.0,
    )
    dynamic_temp_exponent: float | None = RenamedField(
        None,
        renamed_to="dynamic_temp_exponent", original_name="dynatemp_exponent",
        ge=0.0, le=5.0,
    )

    auto_padding: bool | None = RenamedField(
        None,
        description=(
            "When enabled, adds a white space at the begining of the input prompt "
            "if there is no trailing whitespace at the end of the previous action"
        ),
        renamed_to="auto_padding", original_name="frmtadsnsp",
    )
    single_line: bool | None = RenamedField(
        None,
        description="When enabled, removes everything after the first line of the output, including the newline.",
        renamed_to="single_line", original_name="singleline",
    )
    # I want to have a serius word with whoever named these
    merge_consecutive_newlines: bool | None = RenamedField(
        None,
        description=(
            "When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline."
        ),
        renamed_to="merge_consecutive_newlines", original_name="frmtrmblln",
    )
    remove_special_characters: bool | None = RenamedField(
        None,
        description="When enabled, removes #/@%}{+=~|^<> from the output.",
        renamed_to="remove_special_characters", original_name="frmtrmspch",
    )
    remove_unfinished_tail: bool | None = RenamedField(
        None,
        description=(
            "When enabled, removes some characters from the end of the output "
            "such that it doesn't end in the middle of a sentence. "
            "If the output is less than one sentence long, does nothing"
        ),
        renamed_to="remove_unfinished_tail", original_name="frmttriminc",
    )

    tfs: float | None = Field(
        None,
        ge=0.0, le=1.0,
    )
    top_a: float | None = Field(
        None,
        ge=0.0, le=1.0,
    )
    top_k: float | None = Field(
        None,
        ge=0.0, le=100.0,
    )
    top_p: float | None = Field(
        None,
        ge=0.001, le=1.0,
    )
    min_p: float | None = Field(
        None,
        ge=0.001, le=1.0,
    )

    typical: float | None = Field(
        None,
        ge=0.001, le=1.0,
    )
    sampler_order: list[int] | None = Field(
        None,
        # description="A list of integers that determines the order in which the sampling values are applied.",
        description="A list of integers representing the sampler order to be used.",
    )


class TextGenerationRequest(HordeRequest):
    prompt: str = Field(
        description="Prompt to send to KoboldAI .",
    )
    softprompt: str | None = Field(
        default=None,
        description=(
            "Which softpompt needs to be used to service this request "
            "For further information, see https://github.com/KoboldAI/KoboldAI-Client/wiki/Soft-Prompts"
        ),
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            "A list of models that are allowed to fulfill this request. "
            "If none are specified, all models are allowed."
        ),
    )
    disable_batching: bool | None = Field(
        default=None,
        description=(
            "If true, the request will not use batching. "
            "This will allow accurate retrieval of seeds. "
            "This feature is restricted to trusted users and patrons."
        ),
    )
    params: TextGenerationParams | None = None

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

    dry_run: bool | None = Field(
        default=None,
        description="If true, the estimated kudo cost of the request will be returned instead of generated text.",
    )
    allow_downgrade: bool | None = Field(
        default=None,
        description="If true, the request will be allowed to be downgraded in case the upfront kudos cost is not met. ",
    )
    proxied_account: str | None = Field(
        default=None,
        description=(
            "If using a service account as a proxy, "
            "provide this value to identify the actual account from which this request is coming from."
        ),
    )
    # You're on your own for this, this lib won't help you with these
    webhook: str | None = Field(
        default=None,
        description=(
            "Provide a URL where the AI Horde will send a POST call after each delivered generation. "
            "The request will include the details of the job as well as the request ID."
        ),
    )


class GeneratedTextMetadata(HordeModel):
    type: Literal["censorship"] = Field(
        description="The relevance of the metadata field.",
    )
    value: Literal["csam"] | str = Field(
        description="The value of the metadata field.",
    )
    ref: str | None = Field(
        default=None,
        description="A reference for the metadata (e.g. a lora ID).",
        max_length=255,
    )


class FinishedTextGeneration(HordeModel):
    text: str = Field(
        description="The generated text.",
    )

    worker_id: str = Field(
        description="The UUID of the worker which generated this text.",
    )
    worker_name: str = Field(
        description="The name of the worker which generated this text.",
    )
    model: str = Field(
        description="The name of the model which generated this text.",
    )
    seed: int | None = Field(
        default=None,
        description="The seed used to generate this text.",
    )

    censored: bool | None = Field(
        default=None,
        description="When true this image has been censored by the worker's safety filter.",
    )
    gen_metadata: list[GeneratedTextMetadata] | None = Field(
        description="A list of metadata fields for this image.",
    )
    # TODO: When https://github.com/pydantic/pydantic/issues/2255 gets implemented, mark this as deprecated properly
    state: Literal["ok", "censored"] = Field(
        description="DEPRECATED! The state of this generation. Made obsolete by the `gen_metadata` field.",
    )


class TextGenerationStatus(GenerationCheck):
    generations: list[FinishedTextGeneration] = Field(
        description="A list of generated texts.",
    )
