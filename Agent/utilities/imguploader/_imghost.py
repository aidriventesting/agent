import os
from typing import Optional
import requests
from Agent.utilities._logger import RobotCustomLogger
from Agent.config.config import Config
from Agent.utilities.imguploader._imgbase import BaseImageUploader


class FreeImageHostUploader(BaseImageUploader):
    def __init__(self):
        self.config = Config()
        self.base_url = "https://freeimage.host/api/1/upload"
        self.headers = {"Accept": "application/json"}
        self.logger = RobotCustomLogger()

    @property
    def api_key(self):
        api_key = self.config.FREEIMAGEHOST_API_KEY
        if not api_key:
            self.logger.error("FREEIMAGEHOST_API_KEY not found in configuration")
        return api_key

    def _make_request(self, payload: dict, files: bool = False) -> Optional[str]:
        try:
            if files:
                response = requests.post(self.base_url, files=payload)
            else:
                response = requests.post(self.base_url, data=payload, headers=self.headers)
            response.raise_for_status()
            json_data = response.json()
            return self._extract_url(json_data)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"From image host uploader: API Request Failed: {e}")
            return None
        except ValueError:
            self.logger.error("From image host uploader: Invalid JSON response from API")
            return None
        except Exception as e:
            self.logger.error(f"From image host uploader: Other unexpected error: {e}")
            return None

    def _extract_url(self, json_data: dict) -> Optional[str]:
        image_data = json_data.get("image", {})
        return image_data.get("display_url") or image_data.get("url")

    def upload_from_base64(self, base64_data: str, filename: str = "screenshot.png") -> Optional[str]:
        payload = {"key": self.api_key, "action": "upload", "source": base64_data, "format": "json"}
        return self._make_request(payload)

    # def upload_from_file(self, file_path: str) -> Optional[str]:
    #     try:
    #         with open(file_path, "rb") as f:
    #             payload = {
    #                 "key": (None, self.api_key),
    #                 "action": (None, "upload"),
    #                 "format": (None, "json"),
    #                 "source": (os.path.basename(file_path), f),
    #             }
    #             return self._make_request(payload, files=True)
    #     except FileNotFoundError:
    #         full_path = os.path.abspath(file_path)
    #         self.logger.error(f"File not found: {full_path}")
    #         raise FileNotFoundError(f"File not found: {full_path}")


