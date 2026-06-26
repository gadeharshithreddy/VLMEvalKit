from .image_vqa import ImageVQADataset

class VQAv2Dataset(ImageVQADataset):
    DATASET_URL = {
        'VQAv2Dataset': 'https://huggingface.co/datasets/GadeHarshithReddy/VQAv2Val2014/resolve/main/VQAv2Dataset.tsv'
    }

    DATASET_MD5 = {
        'VQAv2Dataset': 'CEFDF3014C0321ABC0B134C27275D0EA'
    }

    # Could work to have --data VQAv2 instead of --data VQAv2Dataset // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQAv2Dataset']
    