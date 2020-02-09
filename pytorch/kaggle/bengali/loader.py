import pandas as pd
import utils
from torch.utils.data import Dataset

class BanDataset(Dataset):
    def __init__(self,
                 DATA_PATHS,
                 LABEL,
                 HEIGHT,
                 WIDTH,
                 SIZE
                 ):
        self.data = DATA_PATHS,
        self.label = pd.read_csv(LABEL[0])
        self.height = HEIGHT,
        self.width = WIDTH,
        self.size = SIZE,
        self.parquet_imageid, self.parquet_image = utils.read_parquet(DATA_PATHS, HEIGHT, WIDTH)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        vowel = self.label.vowel_diacritic.values[idx]
        root = self.label.grapheme_root.values[idx]
        consonant = self.label.consonant_diacritic.values[idx]
        image = self.parquet_image[idx]
        processed_image = self.__data_generation(image)
        return processed_image, root, vowel, consonant

    def __data_generation(self, img):
        # 이미지 크롭, 리사이징
        cropped_img = utils.crop(img)
        resized_img = utils.resize(cropped_img, size=self.size)
        normalized_img = utils.normalize(resized_img)
        return normalized_img


