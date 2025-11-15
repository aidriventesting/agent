from Agent.utilities._logger import RobotCustomLogger


class BBoxToClickCoordinates:
    def __init__(self):
        self.logger = RobotCustomLogger()
        self.logger.info("--------------------------------")
        self.logger.info("Initializing BBoxToClickCoordinates")
        self.logger.info("--------------------------------")

    def _convert_normalized_bbox_to_real_coordinates(self, bbox, screen_width, screen_height):
        self.logger.info("coverting normalized bbox to real coordinates")
        # les coordonnées normalisées rendu par omni parser sont dans l'ordre x1, y1, x2, y2
        x1, y1, x2, y2 = bbox
        
        # le centre de la bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # covertirn en pixels
        real_x = int(center_x * screen_width)
        real_y = int(center_y * screen_height)

        self.logger.info(f"Converted normalized bbox to real coordinates: {real_x}, {real_y}")
        return {'x': real_x, 'y': real_y}
        
    def _get_screen_dimensions(self, driver):
        try:
            screen_size = driver.get_window_size()
            capabilities = driver.capabilities
            device_name = capabilities.get('deviceName')
            self.logger.info(f"device screen size: {screen_size} for device: {device_name}")
            return screen_size['width'], screen_size['height']
        except Exception as e:
            print(f"Erreur lors de la récupération des dimensions de l'écran: {e}")
            return None, None
        
    def get_real_coordinates(self, driver, bbox):
        """
        Convertit les coordonnées normalisées en coordonnées réelles pour le clic.
        
        Args:
            driver: Instance du driver Appium (car on get les dimensions de l'écran de DUT)
            bbox: Liste [x1, y1, x2, y2] représentant la boîte englobante normalisée (0-1)
        """
        self.logger.info(f"Getting real coordinates for bbox: {bbox}")
        width, height = self._get_screen_dimensions(driver)
        self.logger.info("--------------------------------")
        self.logger.info(f"End of BBoxToClickCoordinates")
        self.logger.info("--------------------------------")
        return self._convert_normalized_bbox_to_real_coordinates(bbox, width, height)