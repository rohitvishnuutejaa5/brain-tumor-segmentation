
import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class MRISegmenter:
    """
    A class to handle 3D brain MRI segmentation using a U-Net model.
    """
    def __init__(self, input_shape=(128, 128, 128, 1), data_path='dataset/', output_path='output/'):
        self.input_shape = input_shape
        self.data_path = data_path
        self.output_path = output_path
        self.model = self.build_unet()
        self.compile_model()

    def load_and_preprocess(self, num_samples=10):
        scan_files = sorted([os.path.join(self.data_path, 'scans', f) for f in os.listdir(os.path.join(self.data_path, 'scans'))])
        mask_files = sorted([os.path.join(self.data_path, 'masks', f) for f in os.listdir(os.path.join(self.data_path, 'masks'))])

        scans, masks = [], []
        for i in range(min(num_samples, len(scan_files))):
            scan_img = nib.load(scan_files[i]).get_fdata()
            mask_img = nib.load(mask_files[i]).get_fdata()

            scan_img = scan_img / np.max(scan_img)
            scan_img = tf.image.resize(np.expand_dims(scan_img, axis=-1), self.input_shape[:3])

            mask_img[mask_img > 0] = 1.0
            mask_img = tf.image.resize(np.expand_dims(mask_img, axis=-1), self.input_shape[:3])

            scans.append(scan_img)
            masks.append(mask_img)

        return np.array(scans), np.array(masks)

    def build_unet(self):
        inputs = Input(self.input_shape)
        c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling3D((2, 2, 2))(c1)

        c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling3D((2, 2, 2))(c2)

        c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)

        u4 = UpSampling3D((2, 2, 2))(c3)
        u4 = concatenate([u4, c2])
        c4 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
        c4 = BatchNormalization()(c4)

        u5 = UpSampling3D((2, 2, 2))(c4)
        u5 = concatenate([u5, c1])
        c5 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
        c5 = BatchNormalization()(c5)

        outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c5)

        return Model(inputs=[inputs], outputs=[outputs])

    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=25, batch_size=2):
        print("Loading and preprocessing data...")
        scans, masks = self.load_and_preprocess()
        print("Data loaded. Starting training...")
        self.model.fit(scans, masks, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        self.model.save('brain_unet_model.h5')
        print("Training complete. Model saved.")

    def predict(self):
        if not os.path.exists('brain_unet_model.h5'):
            print("Trained model not found. Please train the model first.")
            return

        self.model.load_weights('brain_unet_model.h5')
        scans, masks = self.load_and_preprocess(num_samples=1)

        if len(scans) == 0:
            print("No data found for prediction.")
            return

        prediction = self.model.predict(scans)
        self.visualize_segmentation(scans[0], masks[0], prediction[0])
        self.save_prediction(prediction[0])

    def visualize_segmentation(self, scan, mask, prediction):
        slice_idx = scan.shape[2] // 2

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(scan[:, :, slice_idx, 0], cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask[:, :, slice_idx, 0], cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Segmentation")
        plt.imshow(prediction[:, :, slice_idx, 0] > 0.5, cmap='gray')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        plt.savefig(os.path.join(self.output_path, 'visualization_example.png'))
        plt.show()

    def save_prediction(self, prediction):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        nifti_img = nib.Nifti1Image((prediction > 0.5).astype(np.uint8), np.eye(4))
        nib.save(nifti_img, os.path.join(self.output_path, 'segmented_mask_example.nii.gz'))
        print(f"Prediction saved to {self.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help="Mode to run: 'train' or 'predict'")
    args = parser.parse_args()

    segmenter = MRISegmenter()
    if args.mode == 'train':
        segmenter.train_model()
    elif args.mode == 'predict':
        segmenter.predict()
