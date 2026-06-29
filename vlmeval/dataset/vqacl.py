from .image_vqa import ImageVQADataset

class VQACL(ImageVQADataset):
    DATASET_URL = {
        'VQACL': 'https://huggingface.co/datasets/GadeHarshithReddy/VQACL/resolve/main/VQACL.tsv'
    }

    DATASET_MD5 = {
        'VQACL': '7593A589336D02EE2C89E0BE80ECFA5E'
    }

    # Could work to have --data VQACL instead of --data VQACL // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQACL']