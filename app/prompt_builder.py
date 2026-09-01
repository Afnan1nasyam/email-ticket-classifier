"""Prompt construction from versioned templates.

`PromptBuilder` loads a prompt template from `prompts/` and renders it with the
category definitions (drawn from `app/config.py`, the single source of truth for
category wording) and the email under classification. Keeping each prompt version
in its own file is what makes "v1 scored X, v3 scored Y" a reproducible claim.

The rendered template is the user message; a short, version-independent role line
is the system prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app import config

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Fixed, version-independent role line. The content we iterate on (v1 -> v3) is
# the template file, not this line.
SYSTEM_PROMPT = (
    "You are a precise email/ticket classification system. Follow the "
    "instructions in the user message exactly and respond with a single JSON "
    "object and nothing else."
)


class PromptBuilder:
    """Load a versioned prompt template and render it for one email."""

    def __init__(
        self,
        prompts_dir: Path = _PROMPTS_DIR,
        categories: dict[str, str] | None = None,
    ) -> None:
        """Create a prompt builder.

        Args:
            prompts_dir: directory containing the template files.
            categories: label -> definition mapping; defaults to
                `config.CATEGORIES`.
        """
        self.prompts_dir = prompts_dir
        self.categories = categories if categories is not None else config.CATEGORIES

    def _render_categories(self) -> str:
        """Render the category definitions as a bulleted list."""
        return "\n".join(f"- {label}: {desc}" for label, desc in self.categories.items())

    def load_template(self, name: str) -> str:
        """Read a template file from the prompts directory.

        Args:
            name: template filename, e.g. ``"v1_zero_shot.txt"``.

        Returns:
            The raw template text.

        Raises:
            FileNotFoundError: if the template does not exist.
        """
        path = self.prompts_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def build(
        self, email_text: str, template: str = "v1_zero_shot.txt"
    ) -> tuple[str, str]:
        """Render a template into a (system_prompt, user_message) pair.

        Args:
            email_text: the (preprocessed) email body to classify.
            template: template filename to load from the prompts directory.

        Returns:
            ``(system_prompt, user_message)`` ready to pass to
            `LLMProvider.complete`.
        """
        template_text = self.load_template(template)
        user_message = template_text.format(
            categories=self._render_categories(),
            labels=", ".join(self.categories),
            email=email_text,
        )
        logger.debug("Built prompt from template %s (%d chars)", template, len(user_message))
        return SYSTEM_PROMPT, user_message
