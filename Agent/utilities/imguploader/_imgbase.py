from abc import ABC, abstractmethod
from typing import Optional


class BaseImageUploader(ABC):
    # @abstractmethod
    # def upload_from_file(self, file_path: str) -> Optional[str]:
    #     pass

    @abstractmethod
    def upload_from_base64(self, base64_data: str) -> Optional[str]:
        pass


