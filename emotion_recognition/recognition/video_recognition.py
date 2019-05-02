from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.applications.nasnet import preprocess_input

graph = tf.get_default_graph()

# parameters for loading data and images
emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'
if (__name__ != '__main__'):
    emotion_model_path = 'emotion_recognition/recognition/'+emotion_model_path
    detection_model_path = 'emotion_recognition/recognition/'+detection_model_path


emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
def readVideo(path):

    return None

def videoRecognition(path,targetPath):
    emotion_window = []
    frame_window = 10
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(targetPath, -1, fps, size)
    success, frame = video.read()
    index = 0
    while success:
        index+=1
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

        for face_coordinates in faces:
            x1, y1, width, height = face_coordinates
            x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
            # x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            global graph
            with graph.as_default():
                emotion_prediction = emotion_classifier.predict(gray_face)
            # emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_text = mode(emotion_window)
            except:
                continue
            color = (0, 0, 255)
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(rgb_image, emotion_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        videoWriter.write(bgr_image)
        success, frame = video.read()
    videoWriter.release()
    video.release()
#videoRecognition("../../media/vdo/test5.mp4","trans.mp4")
