"""AML screenshot contract, including strict inline media validation."""
import base64
import binascii
from io import BytesIO
import re
from typing import Annotated, Literal
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 30 * 1024 * 1024
DATA_URI = re.compile(r"^data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/]*={0,2})$")


def decode_image(uri: str) -> tuple[str, bytes]:
    if len(uri) > (MAX_IMAGE_BYTES * 4 // 3) + 128:
        raise ValueError("single image exceeds 10 MiB")
    match = DATA_URI.fullmatch(uri)
    if not match:
        raise ValueError("only inline JPEG, PNG or WebP Base64 Data URIs are accepted")
    try:
        data = base64.b64decode(match[2], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("invalid Base64 image") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or exceeds 10 MiB")
    try:
        with Image.open(BytesIO(data)) as im:
            expected = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}[match[1]]
            if im.format != expected or im.width * im.height > 40_000_000:
                raise ValueError("image format mismatch or pixel count exceeds 40 million")
            if getattr(im, "n_frames", 1) != 1:
                raise ValueError("animated images are not supported")
            im.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("invalid image payload") from None
    return "image/" + match[1], data


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str
    detail: Literal["low", "high", "auto"] | None = None

    @field_validator("url")
    @classmethod
    def valid_image(cls, v):
        decode_image(v)
        return v


class TextPart(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonblank(cls, v):
        if not v.strip():
            raise ValueError("text part must not be blank")
        return v


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


Part = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = str | list[Part]


def media_size(content) -> int:
    if isinstance(content, str):
        return 0
    size = 0
    for part in content:
        p = part.model_dump(exclude_none=True) if isinstance(part, BaseModel) else part
        if p["type"] == "image_url":
            uri = p["image_url"]["url"]
            encoded = uri.split(",", 1)[1]
            size += len(encoded) * 3 // 4 - len(encoded) + len(encoded.rstrip("="))
    return size


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Content
    timestamp: int | None = Field(default=None, ge=0, le=253402300799000)

    @field_validator("content")
    @classmethod
    def nonempty(cls, v):
        if not v or (isinstance(v, str) and not v.strip()):
            raise ValueError("content must not be empty")
        return v


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=512)
    user_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_media(self):
        if sum(media_size(m.content) for m in self.messages) > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("Add images exceed 30 MiB in total")
        return self


class SearchRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=512)
    query: Content
    options: list[str] | None = Field(default=None, max_length=100)
    top_k: int = Field(ge=1, le=100)

    @field_validator("query")
    @classmethod
    def nonempty(cls, v):
        return Message.nonempty(v)

    @model_validator(mode="after")
    def validate_media(self):
        if media_size(self.query) > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("query images exceed 30 MiB in total")
        return self
