import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np

# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 1
# constants
FONTS = cv.FONT_HERSHEY_COMPLEX

# Puntos de cada ojo
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh
# camara object
camera = cv.VideoCapture('VideoMovOcular.mp4')

# deteccion de la cara
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
          [cv.circle(img, p, 2, (247, 191, 190), -1) for p in mesh_coord]
    return mesh_coord

#calculo de la distancia eucladiana
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

#Calculo del radio al parpadear con ambos ojos
def blinkRatio(img, landmarks, right_indices, left_indices):
    # right eye
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # left eye
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lhDistance = euclaideanDistance(lh_right, lh_left)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance
    ratio = (reRatio + leRatio)/2

    return ratio

#funcion para obtener solo los ojos
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    #cv.imshow('mask', mask)
    eyes = cv.bitwise_and(gray, gray, mask=mask)

    #cv.imshow('eyes', eyes)

    eyes[mask == 0] = 155
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left

#funcion para estimar las pupilas
def positionEstimator(cropped_eye):
    h, w = cropped_eye.shape
    gb = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gb, 3)

    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)
    piece = int(w / 3)

    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    eye_position = pixelCounter(right_piece, center_piece, left_piece)
    return eye_position

#funcion para interpretar en que parte del ojo esta la pupila.
#el ojo se divide en 3 partes para determinar derecha, izquierda y centro
def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)

    print('R:',right_part)
    print('C',center_part)
    print('L',left_part)

    eye_parts = [right_part, center_part, left_part]
    max_index = eye_parts.index(max(eye_parts))
    print(max_index)
    pos_eye = ''
    if max_index == 0:
        pos_eye = 'DERECHA'
        # color = [(0, 0, 0), (0, 255, 0)]
    elif max_index == 1:
        pos_eye = 'CENTRO'
        # color = [(255, 233, 0), (247, 191, 190)]
    elif max_index == 2:
        pos_eye = 'IZQUIERDA'
        # color = [(255, 127, 0), (0, 0, 0)]
    else:
        pos_eye = 'Cerrado'
        # color = [(255, 127, 0), (0, 0, 0)]

    return pos_eye

#en esta programacion no se calibraron los ojos ya que no tuvimos en cuenta frenar y acelarar. El codigo es solo una
#opcion  para el guino derecho y guino izquierdo.
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
             #se dibuja los contornos de los ojos
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0))
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0))
            #se mide el radio del parpadeo
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            if ratio > 3.3:
                #si el radio supera 3.3 es parpadeo. Este valor puede salir de la calibracion de parpadeo
                CEF_COUNTER += 1
                cv.putText(frame, 'Parpadeo', (250, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    #contador de parpadeos
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            cv.putText(frame, f'Parpadeos: {TOTAL_BLINKS}', (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            #tomo las coordenadas de los ojos
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            #tomo solo los ojos de la cara
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            #estimo donde esta la pupila
            eye_position = positionEstimator(crop_right)
            cv.putText(frame, eye_position, (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        cv.imshow('En funcionamiento', frame)
        key = cv.waitKey(15)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
