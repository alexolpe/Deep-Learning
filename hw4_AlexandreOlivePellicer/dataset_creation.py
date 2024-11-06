import cv2
from PIL import Image
from pycocotools.coco import COCO
import os

dataDir='.'

dataTypes=['train2017', 'train2017']
num_samples = [1600, 400]
saving_paths = ['../COCOTraining', '../COCOValidation']

classes = ['boat','couch','dog', 'cake', 'motorcycle']
used_ids = []

k=0
added = 0

# This for iterates 2 times. First to create the training dataset and second to create the training dataset
for i, dataType in enumerate(dataTypes):
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    
    # For each class, we first get all the ids of the images that have been labeled with the specific class
    for cls in classes:
        catIds = coco.getCatIds(catNms=[cls])
        imgIds = coco.getImgIds(catIds=catIds )
        
        # We go across all the ids stored until reaching the desired number of samples for the corresponding class
        while added < num_samples[i]:
            img_name = str(imgIds[k]).zfill(12)
            file_path = f"../../../../Downloads/train2017/train2017/{img_name}.jpg"
            # If the image with specific id hasn't been previously added to either the training dataset or the validation dataset, then we load it, resize it and add it in the corresponding dataset
            if imgIds[k] not in used_ids and os.path.exists(file_path):
                original_image = cv2.imread(file_path)
                
                # Convert the image from BGR to RGB (OpenCV uses BGR by default)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # Read the image using skimage
                resized_image_opencv = cv2.resize(original_image, (64, 64))

                # Save the resized image using PIL
                resized_image_pil = Image.fromarray(resized_image_opencv)
                resized_image_pil.save(f'{saving_paths[i]}/{cls}_{added}_{imgIds[k]}.jpg')
                
                # We add the id of the image we have just added to our dataset to not using it again and increase the counter for added images
                used_ids.append(imgIds[k])
                added +=1
            
            # We move to the next id    
            k += 1
        
        # For each class we reset the index for the array of ids "k" and the counter of added images "added"
        k=0
        added = 0

# If the code has been correctly implemented, the length of used ids should be 1600*5 + 400*5 = 10000            
print(len(used_ids))