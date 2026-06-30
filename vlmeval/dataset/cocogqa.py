from .image_vqa import ImageVQADataset

class COCOGQA(ImageVQADataset):
    DATASET_URL = {
        'COCOGQA': ''
    }

    DATASET_MD5 = {
        'COCOGQA': '6309D13E2F0B2A70F101991641203282'
    }

    # Could work to have --data COCOGQA instead of --data COCOGQA // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['COCOGQA']