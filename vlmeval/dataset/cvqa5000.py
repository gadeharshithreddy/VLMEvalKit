from .image_vqa import ImageVQADataset

class CVQA5000(ImageVQADataset):
    DATASET_URL = {
        'CVQA5000': ''
    }

    DATASET_MD5 = {
        'CVQA5000': 'BD8297DBDDCCEB462234EEBA448D6993'
    }

    # Could work to have --data CVQA5000 instead of --data CVQA5000 // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['CVQA5000']