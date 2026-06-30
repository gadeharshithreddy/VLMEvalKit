from .image_vqa import ImageVQADataset

class VQACP5000(ImageVQADataset):
    DATASET_URL = {
        'VQACP5000': ''
    }

    DATASET_MD5 = {
        'VQACP5000': '8E0C403E688097EE10C7B69971A369F1'
    }

    # Could work to have --data VQACP5000 instead of --data VQACP5000 // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQACP5000']