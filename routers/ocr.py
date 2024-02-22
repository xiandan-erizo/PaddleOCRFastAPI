# -*- coding: utf-8 -*-

from utils.ocrfactory import Ocrf
from fastapi import APIRouter, HTTPException, UploadFile, status
from models.OCRModel import *
from models.RestfulModel import *
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray, convert_numpy
import requests

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.get('/{model_name}/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(model_name: str, image_path: str):
    ocr = Ocrf.get(model_name)
    result = ocr.perform_ocr(image_path=image_path)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/{model_name}/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(model_name: str, base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    ocr = Ocrf.get(model_name)
    result = ocr.perform_ocr(image_path=img)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/{model_name}/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(model_name: str, file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png", ".jpeg")):  # 只处理常见格式图片
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        ocr = Ocrf.get(model_name)
        result = ocr.perform_ocr(image_path=img)
        result = convert_numpy(result)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel


@router.get('/{model_name}/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(model_name: str, imageUrl: str):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    # 只处理常见格式图片 (jpg / png)
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        ocr = Ocrf.get(model_name)
        result = ocr.perform_ocr(image_path=img)
        restfulModel.data = result
        restfulModel.message = "Success"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel
