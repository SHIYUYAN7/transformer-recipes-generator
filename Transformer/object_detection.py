import torch
from ultralytics import YOLO
from IPython.display import display, Image
from roboflow import Roboflow
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms


rf = Roboflow(api_key="xWqeDWmyJcSHK3O2sjcC")
project = rf.workspace("project-vpjcp").project("test-xwxsh")
version = project.version(10)
dataset = version.download("yolov8")


def detect_object(image_path):
    trained_model = YOLO('best.pt')
    res=trained_model.predict(
        source=image_path,
        conf=0.5,
        save=True # save to file or not
    )
    return res


def get_class_and_crop_image(image_path):
    res = detect_object(image_path)
    img = Image.open(image_path)
    cropped_images = []

    for _, box in enumerate(res[0].boxes.xyxy):
        left, top, right, bottom = map(int, box[:4])
        cropped_image = img.crop((left, top, right, bottom))
        # print(f"Displaying cropped image {i}:")
        # display(cropped_image)
        cropped_images.append(cropped_image)
    
    ingredients = res[0].boxes.cls
    return ingredients, cropped_images

resnet = models.resnet18(pretrained=True)
resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
resnet.fc = torch.nn.Identity()

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_features(image_path):
    ingredients, cropped_images = get_class_and_crop_image(image_path)
    features = []
    resnet.eval()
    with torch.no_grad():
        for img in cropped_images:
            img = transform(img).unsqueeze(0)
            feature = resnet(img)
            feature = feature.view(feature.size(0), -1)
            features.append(feature)
    features = torch.cat(features, dim=0)
    return ingredients, features


ingredients, features = get_image_features('../test_images/test5.jpg')
print(ingredients, features.shape)