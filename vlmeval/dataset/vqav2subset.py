from .image_vqa import ImageVQADataset

class VQAv2DatasetSubset(ImageVQADataset):
    DATASET_URL = {
        'VQAv2DatasetSubset': 'https://huggingface.co/datasets/GadeHarshithReddy/VQAv2Val2014/resolve/main/vqav2_val_subset.tsv'
    }

    DATASET_MD5 = {
        'VQAv2DatasetSubset': '2AB8317DC496F213DCFDB0AA2AAED26C'
    }

    # Could work to have --data VQAv2 instead of --data VQAv2Dataset // Further experimentation needed
    @classmethod
    def supported_datasets(cls):
        return ['VQAv2DatasetSubset']