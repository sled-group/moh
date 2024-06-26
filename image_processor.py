from PIL import Image

class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        """
        Preprocess an image by converting it to RGB if necessary.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Preprocessed image.
        """
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return image