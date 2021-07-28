import numpy as np
import os
import cv2


def createCleanBG(input_path, memorize, skipping, save_result, output_path=None):
    """
    this function generates a clean background image from a video sequence
    Parameters:
        input_path (string): path of input video sequence
        memorize (int): number of frames that are used to generate a clean background
        skipping (int): number of frames to skip before caching another frame to generate the clean background
        save_result (boolean): boolean parameter to decide whether to save the generated result or not
        output_path (string): output path of the generated clean background
    Returns:
        generated clean background image
    """
    cap = cv2.VideoCapture(input_path)
    frameNum = 0
    num_of_memorized_frames = 0
    frames = []
    while num_of_memorized_frames < memorize:
        ret, frame = cap.read()
        if ret:
            if frameNum % skipping == 0:
                frames.append(frame)
                num_of_memorized_frames += 1
            frameNum += 1
        else:
            break

    cleanPlate = np.median(np.array(frames), axis=0).astype(np.uint8)
    if save_result:
        cv2.imwrite(os.path.join(output_path, 'cleanBG.jpg'), cleanPlate)
    return cleanPlate


def preProcess(img, adjust_vibrance=True, vibrance_amount=45,
               adjust_shadows=True, brighten_amount=45):
    """
    this function applies a defined color adjustments to a single frame
    Parameters:
        img (ndarray): input image in BGR color space to apply adjustments on
        adjust_vibrance (boolean): boolean parameter to decide whether to use vibrance adjustment or not
        vibrance_amount (int): increasing vibrance amount
        adjust_shadows (boolean): boolean parameter to decide whether to use shadows adjustment or not
        brighten_amount (int): increasing brightness amount
    Returns:
         image in BGR color space after applying adjustments
    """
    if adjust_vibrance or adjust_shadows:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if adjust_vibrance:
        img = vibrance(img, vibrance_amount)
    if adjust_shadows:
        img = brightenShadows(img, brighten_amount)

    if adjust_vibrance or adjust_shadows:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def vibrance(img, amount):
    """
    this function is an approximated implementation for vibrance filter by Photoshop that increases the saturation of
    an image in a way that the increasing amount for the low saturated pixels is more than the increasing amount for
    pixels that are already saturated
    Parameters:
        img (ndarray): input image in HSV color space
        amount (int): increasing vibrance amount
    Returns:
         image in HSV color space after applying vibrance filter
    """
    amount = min(amount, 100)
    sat_increase = ((255 - img[:, :, 1]) / 255 * amount).astype(np.uint8)
    img[:, :, 1] += sat_increase
    return img


def brightenShadows(img, amount):
    """
    this function increases the brightness of the dark pixels of an image
    Parameters:
        img (ndarray): input image in HSV color space
        amount (int): increasing brightness amount
    Returns:
         image in HSV color space after applying brightness filter
    """
    amount = min(amount, 100)
    val_inc = ((255 - img[:, :, 2]) / 255 * amount).astype(np.uint8)
    img[:, :, 2] += val_inc
    return img


def segment(cleanBG, input_path, output_path):
    """
    this function generates mask images for the input video sequence with the help of the clean background image
    and writes the generated mask images to the defined output path
    Parameters:
        cleanBG (ndarray): clean background image that is generated from input video sequence
        input_path (string): path of input video sequence
        output_path (string): output path of the generated mask images
    """
    cleanBG = preProcess(cleanBG,
                         adjust_vibrance=True, vibrance_amount=40,
                         adjust_shadows=True, brighten_amount=45)
    cleanBG = cleanBG.astype(np.int16)
    cleanBG = cv2.GaussianBlur(cleanBG, (3, 3), 0)

    area_tresh = cleanBG.shape[0] * cleanBG.shape[1] * 0.0001
    cap = cv2.VideoCapture(input_path)
    frameNum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = preProcess(frame,
                               adjust_vibrance=True, vibrance_amount=40,
                               adjust_shadows=True, brighten_amount=45)
            frame = frame.astype(np.int16)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            sub = np.subtract(cleanBG, frame)
            abs = np.abs(sub)
            channels_sum = (np.sum(abs, axis=2) / 3)
            channels_sum = channels_sum.astype(np.uint8)

            mask = cv2.threshold(channels_sum, 23, 255, cv2.THRESH_BINARY)[1]
            mask = mask.astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            cleanedMask = np.zeros(cleanBG.shape[0:2]).astype(np.uint8)
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > area_tresh:
                    cv2.drawContours(cleanedMask, [contours[i]], -1, 255, cv2.FILLED)

            imgName = 'seq_' + '{:03d}'.format(frameNum) + '.jpg'
            cv2.imwrite(os.path.join(output_path, imgName), cleanedMask)
            frameNum += 1
        else:
            break
