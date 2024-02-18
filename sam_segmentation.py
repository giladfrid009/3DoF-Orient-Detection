from segment_anything import SamPredictor, sam_model_registry
import numpy as np


class SAMSegmentation:
    def __init__(self, device: str = "auto"):
        self.download_weights_if_needed()
        sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
        self._predictor = SamPredictor(sam)
        self._predictor.model.to(device)

    def segment_center(self, image: np.ndarray):
        h, w = image.shape[:2]
        point_coords = np.array([[w / 2, h / 2]])
        point_labels = np.array([1])

        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best_mask = np.argmax(scores)
        return masks[best_mask], scores[best_mask]

    def download_weights_if_needed(self):
        import os
        import urllib.request

        if os.path.exists("./models/sam_vit_b_01ec64.pth"):
            return

        print("Downloading SAM weights...")
        # make models dir if needed:
        os.makedirs("./models", exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        # download, original file name but to models dir:
        urllib.request.urlretrieve(url, "./models/sam_vit_b_01ec64.pth")
        print("Done.")
