from image_vqa import ImageVQADataset

class VQAv2Dataset(ImageVQADataset):
    DATASET_URL = {
        'VQAv2Dataset': ''
    }

    DATASET_MD5 = {
        'VQAv2Dataset': ''
    }

    # Could work to have --data VQAv2 instead of --data VQAv2Dataset // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQAv2Dataset']
    