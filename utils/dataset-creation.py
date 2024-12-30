import os

import cv2
from utils import config
from bs4 import BeautifulSoup
from imutils import paths
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def apply_gaussian():
    for img_name in os.listdir("../gauss"):
        img_path = os.path.join("../gauss", img_name)
        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imwrite('test/ajde/' + img_name, img)


def augment_data():
    # apply_gaussian()
    # exit(0)
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rotation_range=15, brightness_range=(0.7, 1.4),
                                       shear_range=0.1, zoom_range=[0.95, 1.25], horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory("test", batch_size=20, class_mode=None,
                                                        target_size=(224, 224), save_to_dir="../train/ajde", save_prefix='aug-',
                                                        save_format='jpeg')
    i = 0
    for image in train_generator:
        i += 1
        if i > 1044:
            break


if __name__ == '__main__':
    for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

    imagePaths = list(paths.list_images(config.ORIG_IMAGES))

    totalPositive = 0
    totalNegative = 0

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}...".format(i + 1,
                                                        len(imagePaths)))

        filename = imagePath.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]
        annotPath = os.path.sep.join([config.ORIG_ANNOTS,
                                      "{}.xml".format(filename)])

        contents = open(annotPath).read()
        soup = BeautifulSoup(contents, "html.parser")
        gtBoxes = []

        w = int(soup.find("width").string)
        h = int(soup.find("height").string)

        for o in soup.find_all("object"):
            label = o.find("name").string
            xMin = int(o.find("xmin").string)
            yMin = int(o.find("ymin").string)
            xMax = int(o.find("xmax").string)
            yMax = int(o.find("ymax").string)

            xMin = max(0, xMin)
            yMin = max(0, yMin)
            xMax = min(w, xMax)
            yMax = min(h, yMax)

            gtBoxes.append((xMin, yMin, xMax, yMax))

        image = cv2.imread(imagePath)

        positiveROIs = 0

        for gtBox in gtBoxes:
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            roi = None
            outputPath = None

            roi = image[gtStartY:gtEndY, gtStartX:gtEndX]
            filename = "{}.png".format(totalPositive)
            outputPath = os.path.sep.join([config.POSITVE_PATH,
                                           filename])

            positiveROIs += 1
            totalPositive += 1

            if roi is not None and outputPath is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS,
                                 interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)
