# python
import io
import json
import logging
import torchvision
import torchvision.transforms as transforms
from ultralytics import YOLO
import torch
# 3rdparty
import cv2
import pydantic
import numpy as np

from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image

# project
from datacontract.service_config import ServiceConfig
from datacontract.service_output import ServiceOutput

def load_classifier():
    model=torch.load("resnet50_best_loss.pth")
    return model

# датакласс конфига сервиса

# датакласс выхода сервиса
class ServiceConfig(pydantic.BaseModel):
    name_of_classifier: str
    path_to_classifier: str
    name_of_detector: str
    path_to_detector: str

class Object(pydantic.BaseModel):
    xtl: int
    ytl: int
    xbr: int
    ybr: int
    class_name: str

class ServiceOutput(pydantic.BaseModel):
    objects: list[Object]

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

app = FastAPI()

service_config_path = "configs\\service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)

service_config_adapter = pydantic.TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_json)

# инициализация сетей
class_names = {0: "aircraft", 1: "ship"}
classifier = load_classifier()
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if service_config_python.name_of_detector.lower() == "yolov8":
    detector =YOLO(r"runs\detect\train2\weights\best.pt")
else:
    raise Exception()

logging.info(f"Загружен классификатор {service_config_python.name_of_classifier}")
logging.info(f"Файл весов классификатора: {service_config_python.path_to_classifier}")

logger.info(f"Загружена конфигурация сервиса по пути: {service_config_path}")


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    """Точка доступа для проверки жизни сервиса

    Возрващает:
        HTTP Статус код (ОК)
    """
    return '{"Status" : "OK"}'


@app.post("/file/")
async def inference(image: UploadFile = File(...)) -> JSONResponse:
    """_summary_

    Args:
        image (UploadFile, optional): _description_. Defaults to File(...).

    Returns:
        JSONResponse: _description_
    """
    image_content = await image.read()
    cv_image = np.array(Image.open(io.BytesIO(image_content)))

    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    # создаете объект выхода сервиса
    output_dict = {"objects": []}

    # выполнение детектора
    detector_outputs = detector(cv_image)
    service_output_list = {"objects":[]}
    for res in detector_outputs:
        boxes=res.boxes
        for box in boxes:
            resized_image = cv2.resize(cv_image, (256, 256))
            normalized_image = resized_image.astype(np.float32) / 255.0
            tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = classifier(tensor_image)
            _, predicted_idx = torch.max(output, 1)
            class_index = predicted_idx.item()
            predicted_class = class_names[class_index]
            cords=box.xyxy
            xtl, ytl, xbr, ybr =int(cords[0][0]), int(cords[0][1]), int(cords[0][2]), int(cords[0][3])
            logger.info(f"Принято {xtl, ytl, xbr, ybr, predicted_class}")
            '''crop_object = cv_image[ytl:ybr, xtl:xbr]
            crop_tensor = transform(crop_object)
            class_id = classifier.inference(crop_tensor)
            class_name = class_names[class_id]'''
            service_output_list["objects"].append(Object(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, class_name=predicted_class))
            #output_dict["objects"].append({"xtl": xtl, "xbr": xbr, "ytl": ytl, "ybr": ybr, "class_name": predicted_class})

    '''for box in detector_outputs:
        xtl, ytl, xbr, ybr = box[0], box[1], box[2], box[3]
        crop_object = cv_image[ytl:ybr, xtl:xbr]
        crop_tensor = transform(crop_object)
        class_id = classifier.inference(crop_tensor)
        class_name = class_names[class_id]
        output_dict["objects"].append({"xtl": xtl, "xbr": xbr, "ytl": ytl, "ybr": ybr, "class_name": class_name})'''

    for i in service_output_list:
        logger.info(f"Принято {i}")
    #service_output = ServiceOutput(**output_dict)
    # заполнение service_output

    #service_output_json = service_output.model_dump(mode="json")
    #return JSONResponse(content=jsonable_encoder(service_output_json))
    service_output=ServiceOutput(objects=service_output_list["objects"])
    service_output_json = service_output.model_dump(mode="json")
    return JSONResponse(content=jsonable_encoder(service_output_json))
