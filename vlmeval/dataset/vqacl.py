from .image_vqa import ImageVQADataset

class VQACL(ImageVQADataset):
    DATASET_URL = {
        'VQACL': 'https://huggingface.co/datasets/GadeHarshithReddy/VQACL/resolve/main/VQACL.tsv'
    }

    DATASET_MD5 = {
        'VQACL': 'CBE6A927F5ACAE5C6FBD8559A81964A4'
    }

    # Could work to have --data VQACL instead of --data VQACL // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQACL']