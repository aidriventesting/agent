from typing import Optional
from Agent.config.config import Config
from Agent.utilities.imguploader._imgbb import ImgBBUploader
from Agent.utilities.imguploader._imghost import FreeImageHostUploader
from Agent.utilities.imguploader._imgbase import BaseImageUploader
from Agent.utilities._logger import RobotCustomLogger


class ImageUploader:
    """
    Handles 3 fallback cases with warnings:
    ✅ No provider configured → returns base64 + warning
    ✅ Upload fails (returns None) → returns base64 + warning
    ✅ Exception raised → returns base64 + warning
    """
    def __init__(self, service: str = "auto"):
        self.config = Config()
        self.logger = RobotCustomLogger()
        self.uploader: Optional[BaseImageUploader] = self._select_uploader(service)

    def upload_from_base64(self, base64_data: str) -> Optional[str]:
        """
        Attempts to upload the image. If no provider is configured or if the upload fails,
        returns the image in base64 with a warning.
        """
        # If no uploader is configured, return the base64
        if self.uploader is None:
            self.logger.warning(
                "Fallback: returning the image in base64 (no provider configured)",
                robot_log=False
            )
            return f"data:image/png;base64,{base64_data}"
        
        # Attempt to upload with the configured provider
        try:
            result = self.uploader.upload_from_base64(base64_data)
            
            # If the upload fails (returns None), use the fallback
            if result is None:
                self.logger.warning(
                    "Fallback: returning the image in base64 (upload failed)",
                    robot_log=False
                )
                return f"data:image/png;base64,{base64_data}"
            
            return result
            
        except Exception as e:
            # In case of an unexpected error, log and return the base64
            self.logger.warning(
                f"Fallback: returning the image in base64 (error: {str(e)})",
                robot_log=False
            )
            return f"data:image/png;base64,{base64_data}"

    # def upload_from_file(self, file_path: str) -> Optional[str]:
    #     return self.uploader.upload_from_file(file_path)

    # ----------------------- Internals -----------------------
    def _select_uploader(self, service: str) -> Optional[BaseImageUploader]:
        """Selects an uploader if available, otherwise returns None"""
        if service == "imgbb" or (service == "auto" and self.config.IMGBB_API_KEY):
            return ImgBBUploader()
        elif service == "freeimagehost" or (service == "auto" and self.config.FREEIMAGEHOST_API_KEY):
            return FreeImageHostUploader()
        else:
            self.logger.warning(
                "No upload service configured. Images will be returned in base64.",
                robot_log=True
            )
            return None

