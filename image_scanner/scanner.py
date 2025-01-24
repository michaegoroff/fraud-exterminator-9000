from glob import glob
import cv2
import numpy as np
import scipy



class Scanner:
    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path

    def scan(self,image_name):
        #scan image and grayscale it
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #get variance, skewness, curtosis, entropy
        image_norm = image.astype('float32') / 255.0
        variance = np.var(image_norm)
        skewness = np.mean(scipy.stats.skew(image_norm))
        kurtosis = np.mean(scipy.stats.kurtosis(image_norm))
        entropy = np.mean(scipy.stats.entropy(image_norm))
        return variance, skewness, kurtosis, entropy

    def process_batch(self):
        test_samples = []
        class_labels = []
        #scan .jpg
        for image in glob(f'predict/{self.image_folder_path}/*.jpg'):
            v, s, k, e = self.scan(image)
            test_sample = np.array([v, s, k, e])
            test_samples.append(test_sample)
            if 'fake' in image:
                class_labels.append(1)
            elif 'legit' in image:
                class_labels.append(0)

        #scan .png
        for image in glob(f'predict/{self.image_folder_path}/*.png'):
            v, s, k, e = self.scan(image)
            test_sample = np.array([v, s, k, e])
            test_samples.append(test_sample)
            if 'fake' in image:
                class_labels.append(1)
            elif 'legit' in image:
                class_labels.append(0)

        return test_samples, class_labels

