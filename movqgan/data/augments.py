import imageio
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.parameters as iap
import cv2
import numpy as np
import random

def gen_matrix(image, nb_channels, random_state):
    matrix_A = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])

    matrix_B = np.array([[0, 0, 0],
                         [0, -4, 1],
                         [0, 2, 1]])
    if random_state.rand() < 0.5:
        return [matrix_A] * nb_channels
    else:
        return [matrix_B] * nb_channels

def load_aug(mask_only=False):
    if mask_only == False:
        arithmetic = [
            iaa.Add((-40, 40), per_channel=0.5),
            iaa.AddElementwise((-40, 40), per_channel=0.5),
            # iaa.AdditiveGaussianNoise(scale=0.2 * 255, per_channel=True),
            # iaa.AdditiveLaplaceNoise(scale=0.2 * 255, per_channel=True),
            # iaa.AdditivePoissonNoise(lam=40, per_channel=True),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            # iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            # iaa.Dropout2d(p=0.5, nb_keep_channels=0),
            iaa.ReplaceElementwise(0.1, iap.Normal(128, 0.4 * 128), per_channel=0.5),
            # iaa.ImpulseNoise(0.1),
            # iaa.SaltAndPepper(0.1, per_channel=True),
            iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1), per_channel=True),
            iaa.Salt(0.1),
            # iaa.Pepper(0.1),
            iaa.Invert(0.25, per_channel=0.5),
            iaa.JpegCompression(compression=(70, 99)),
        ]
        blend = [
            iaa.BlendAlphaMask(
                iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
                iaa.Clouds()
            ),
            # iaa.BlendAlphaElementwise(
            #     (0.0, 1.0),
            #     foreground=iaa.Add(100),
            #     background=iaa.Multiply(0.2)),
            # iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13)),
            # iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0)),
            iaa.BlendAlphaHorizontalLinearGradient(
                iaa.TotalDropout(1.0),
                min_value=0.2, max_value=0.8),
            iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
                                       foreground=iaa.AddToHue((-100, 100))),
        ]
        blur = [
            iaa.GaussianBlur(sigma=(0.0, 0.8)),
            # iaa.AverageBlur(k=(2, 11)),
            # iaa.MedianBlur(k=(3, 7)),
        ]
        color = [
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(
                    0,
                    iaa.Add((0, 50))
                )
            ),
            iaa.WithBrightnessChannels(
                iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV]),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.AddToBrightness((-30, 30)),
            iaa.WithHueAndSaturation(
                iaa.WithChannels(0, iaa.Add((0, 50)))
            ),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.MultiplySaturation((0.5, 1.5)),
            iaa.RemoveSaturation(),
            # iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.AddToHue((-50, 50)),
            # iaa.AddToSaturation((-50, 50)),
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((50, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
            ]),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.ChangeColorTemperature((1100, 10000)),
            iaa.UniformColorQuantization(),
        ]
        contrast = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.SigmoidContrast(
                gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
            # iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
            iaa.LinearContrast((0.4, 1.6), per_channel=True),
            iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization()),
            iaa.HistogramEqualization(
                from_colorspace=iaa.HistogramEqualization.BGR,
                to_colorspace=iaa.HistogramEqualization.HSV),
        ]
        edges = [
            iaa.Canny(),
            iaa.Canny(alpha=(0.0, 0.5)),
            iaa.Canny(
                alpha=(0.0, 0.5),
                colorizer=iaa.RandomColorsBinaryImageColorizer(
                    color_true=255,
                    color_false=0
                )
            ),
            iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])
        ]
        imgcorruptlike = [
            iaa.imgcorruptlike.GaussianNoise(severity=2),
            iaa.imgcorruptlike.ShotNoise(severity=2),
            iaa.imgcorruptlike.ImpulseNoise(severity=2),
            iaa.imgcorruptlike.SpeckleNoise(severity=2),
            iaa.imgcorruptlike.GaussianBlur(severity=2),
            # iaa.imgcorruptlike.DefocusBlur(severity=2),
            iaa.imgcorruptlike.Fog(severity=2),
            iaa.imgcorruptlike.Frost(severity=2),
            iaa.imgcorruptlike.Snow(severity=2),
            iaa.imgcorruptlike.Spatter(severity=2),
            # iaa.imgcorruptlike.Contrast(severity=2),
            iaa.imgcorruptlike.Brightness(severity=2),
            iaa.imgcorruptlike.Saturate(severity=2),
            iaa.imgcorruptlike.JpegCompression(severity=2),
            iaa.imgcorruptlike.JpegCompression(severity=2),
            iaa.imgcorruptlike.Pixelate(severity=2),
            iaa.imgcorruptlike.ElasticTransform(severity=2),
        ]
        weather = [
            # iaa.FastSnowyLandscape(
            #     lightness_threshold=(100, 255),
            #     lightness_multiplier=(1.0, 4.0)
            # ),
            iaa.Clouds(),
            iaa.Fog(),
            iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
            iaa.Rain(drop_size=(0.10, 0.20)),
        ]

        augs = [
            arithmetic,
            blend,
            blur,
            color,
            contrast,
            edges,
            imgcorruptlike,
            weather
        ]

        return augs
    else:
        # aug = iaa.CoarseDropout((0.29, 0.3), size_percent=(0.02, 0.25))
        aug = iaa.CoarseDropout((0.29, 0.3), size_percent=0.02)
        return aug


if __name__ == '__main__':
    for _ in range(20):
        image = imageio.imread("a.jpg", pilmode='RGB')

        img_aug = aug(image)

        ia.imshow(img_aug)
