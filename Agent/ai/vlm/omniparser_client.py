from __future__ import annotations

import base64
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from gradio_client import Client, handle_file

from Agent.config.config import Config
from robot.api import logger

class OmniParserError(RuntimeError):
    """Base exception raised when the OmniParser Hugging Face space fails."""


class OmniParserClient:
    """
    Thin wrapper around the OmniParser Hugging Face space.

    Responsibilities:
      - Prepare the image input (local path, remote URL, or base64 string)
      - Merge default parameters with call-specific overrides
      - Execute the prediction request
      - Return the raw API response (image_payload, response_text)
    """

    def __init__(
        self,
        space_id: Optional[str] = None,
        api_name: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.space_id = space_id or Config.OMNIPARSER_SPACE
        self.api_name = api_name or Config.OMNIPARSER_API_NAME
        self.hf_token = hf_token or Config.get_huggingface_token() or None

        client_kwargs: Dict[str, Any] = {}
        if self.hf_token:
            client_kwargs["hf_token"] = self.hf_token

        try:
            self._client = Client(self.space_id, **client_kwargs)
            logger.debug(f"Initialized OmniParser client for space '{self.space_id}'")
        except Exception as exc:  # pragma: no cover - network failure
            raise OmniParserError(f"Failed to initialise OmniParser client: {exc}") from exc

    @contextmanager
    def _resolve_image_source(
        self,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_name: Optional[str] = None,
    ) -> Iterator[Any]:
        """
        Resolve the image input into a gradio `handle_file` payload.

        Yields
        ------
        handle_file(...) result
        """
        temp_path: Optional[str] = None
        try:
            if image_path:
                logger.debug(f"Using local image path for OmniParser: {image_path}")
                yield handle_file(image_path)
                return

            if image_url:
                logger.debug(f"Using remote image URL for OmniParser: {image_url}")
                yield handle_file(image_url)
                return

            if image_base64:
                decoded = base64.b64decode(image_base64)
                suffix = self._infer_suffix(image_name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(decoded)
                    temp_path = tmp_file.name
                self.logger.debug(f"Created temporary image file for OmniParser: {temp_path}")
                yield handle_file(temp_path)
                return

            raise OmniParserError("No image source provided for OmniParser request.")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Deleted temporary image file: {temp_path}")
                except OSError:
                    pass

    @staticmethod
    def _infer_suffix(image_name: Optional[str]) -> str:
        if not image_name:
            return ".png"
        _, ext = os.path.splitext(image_name)
        return ext if ext else ".png"

    def _merge_params(
        self,
        *,
        box_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_paddleocr: Optional[bool] = None,
        imgsz: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = Config.get_omniparser_params(
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        logger.debug(f"OmniParser parameters: {params}")
        return params

    def parse_image(
        self,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_name: Optional[str] = None,
        box_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_paddleocr: Optional[bool] = None,
        imgsz: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Run OmniParser on the provided image source and return the raw API response.

        Returns
        -------
        Tuple[str, str]
            (image_temp_path, parsed_text)
            - image_temp_path: Path to temporary .webp file created by Gradio
            - parsed_text: Raw text containing detected UI elements
        """
        params = self._merge_params(
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )

        with self._resolve_image_source(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_name=image_name,
        ) as image_input:
            try:
                response = self._client.predict(
                    image_input=image_input,
                    api_name=self.api_name,
                    **params,
                )
            except Exception as exc:  # pragma: no cover - network failure
                raise OmniParserError(f"OmniParser request failed: {exc}") from exc

        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise OmniParserError(f"Unexpected OmniParser response format: {response}")

        image_temp_path, parsed_text = response
        image_path_result = image_temp_path or ""
        text_result = parsed_text or ""
        
        logger.debug(f"OmniParser temporary image: {image_path_result}")
        logger.debug(f"OmniParser parsed text: {len(text_result)} bytes")
        
        return image_path_result, text_result


#quick test
if __name__ == "__main__":
    client = OmniParserClient()
    image_temp_path, response_text = client.parse_image(
        image_path="tests/_data/images/screenshots/screenshot-Google Pixel 5-11.0.png"
    )
    print("=" * 80)
    print(f"{image_temp_path}")
    print("=" * 80)
    print(f"{response_text}")