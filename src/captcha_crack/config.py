from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAPTCHA_CRACK_",
        env_file=".env",
        extra="ignore",
    )

    backend: str = "template"
    fallback_backends: str = ""
    hybrid: bool = False

    expected_length: int | None = None
    max_candidates: int = 5

    template_dir: Path = Path("data/templates")

    adaptive_threshold: bool = True
    strip_lines: bool = True
    min_character_area: int = 20

    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    gemini_model: str = "gemini-1.5-flash"

    confidence_floor: float = 0.55

    @property
    def fallback_list(self) -> list[str]:
        if not self.fallback_backends:
            return []
        return [b.strip() for b in self.fallback_backends.split(",") if b.strip()]

    def backend_kwargs(self, name: str) -> dict:
        if name == "template":
            return {"template_dir": self.template_dir}
        if name == "claude":
            return {"model_id": self.bedrock_model, "region": self.aws_region}
        if name == "gemini":
            return {
                "model_name": self.gemini_model,
                "project": self.gcp_project,
                "location": self.gcp_location,
            }
        if name == "rekognition":
            return {"region": self.aws_region}
        return {}
