from keras import backend as K
from keras.models import load_model
import numpy as np
from imutils import face_utils
import time
import dlib
import cv2
import face_recognition
from tensorflow import compat, keras
from os import system, name, listdir
from playsound import playsound
from keyboard import is_pressed as pressed
from threading import Thread
num_cores = 4
num_CPU = 1
num_GPU = 0
config = compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True, device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = compat.v1.Session(config=config)
K.set_session(session)
class FacialLandMarksPosition:
	"""
	The indices points to the various facial features like left ear, right ear, nose, etc.,
	that are mapped from the Facial Landmarks used by dlib's FacialLandmarks predictor.
	"""
	left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
facial_landmarks_predictor = './models/68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)
model = load_model('./models/weights.149-0.01.hdf5')
def predict_eye_state(model, image):
	image = cv2.resize(image, (20, 10))
	image = image.astype(dtype=np.float32)
	image_batch = np.reshape(image, (1, 10, 20, 1))
	image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
	return np.argmax( model.predict(image_batch, verbose=0)[0] )
cap = cv2.VideoCapture(0)
scale = 0.5
face_detected = False
left_eye_open = False
right_eye_open = False
def close():
	from sys import exit
	cap.release()
	cv2.destroyAllWindows()
	exit()
running = True
time_set_face = False
close_time_face = 0
time_set_eyes = False
close_time_eyes = 0
for file in listdir("."):
	if file.startswith("sound"):
		try:
			from sys import _MEIPASS
			base_path = _MEIPASS
		except Exception:
			from os.path import abspath, join
			base_path = abspath(".")
		sound = join(base_path, f"./{file}")
playing = False
musicthread = None
def loopsound(sound):
	global playing
	playsound(sound, True)
	playing = False
system('cls' if name == 'nt' else 'clear')
while running:
	curtime = round(time.time()-0.5)
	# Capture frame-by-frame
	ret, frame = cap.read()
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	original_height, original_width = image.shape[:2]
	resized_image = cv2.resize(image,  (0, 0), fx=scale, fy=scale)
	lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
	l, _, _ = cv2.split(lab)
	resized_height, resized_width = l.shape[:2]
	height_ratio, width_ratio = original_height / resized_height, original_width / resized_width
	face_locations = face_recognition.face_locations(l, model='hog')
	if len(face_locations):
		face_detected = True
		top, right, bottom, left = face_locations[0]
		x1, y1, x2, y2 = left, top, right, bottom
		x1 = int(x1 * width_ratio)
		y1 = int(y1 * height_ratio)
		x2 = int(x2 * width_ratio)
		y2 = int(y2 * height_ratio)
		# draw face rectangle
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
		face_landmarks = face_utils.shape_to_np(shape)
		left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index: FacialLandMarksPosition.left_eye_end_index]
		(x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
		left_eye = gray[y:y + h, x:x + w]
		right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index: FacialLandMarksPosition.right_eye_end_index]
		(x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
		right_eye = gray[y:y + h, x:x + w]
		left_eye_open = True if predict_eye_state(model=model, image=left_eye) else False 
		right_eye_open = True if predict_eye_state(model=model, image=right_eye) else False
		if left_eye_open == True and right_eye_open == True:
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		else:
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
	else:
		face_detected = False
	cv2.imshow('Detection', cv2.flip(frame, 1))
	#if both eyes are closed or the face is missing for 5s, play the sound named "sound.[file extension]" until the spacebar is pressed
	#when the spacebar is pressed, it stops looping and the face/eye detection is turned back on
	if not face_detected:
		if not time_set_face:
			close_time_face = round(time.time()-0.5)
			time_set_face = True
		if curtime-close_time_face >= 5:
			while not pressed("space"):
				if not playing:
					playing = True
					Thread(target=loopsound, args=(sound,), daemon=True).start()
				ret, frame = cap.read()
				cv2.imshow('Detection', cv2.flip(frame, 1))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					running = False
					close()
			close_time_face = 0
			time_set_face = False
	else:
		time_set_face = False
	if not left_eye_open:
		if not time_set_eyes:
			close_time_eyes = round(time.time()-0.5)
			time_set_eyes = True
		if curtime-close_time_eyes >= 5 and not right_eye_open:
			while not pressed("space"):
				if not playing:
					playing = True
					Thread(target=loopsound, args=(sound,), daemon=True).start()
				ret, frame = cap.read()
				cv2.imshow('Detection', cv2.flip(frame, 1))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					running = False
					close()
			close_time_eyes = 0
			time_set_eyes = False
	else:
		time_set_eyes = False
	if not right_eye_open:
		if not time_set_eyes:
			close_time_eyes = round(time.time()-0.5)
			time_set_eyes = True
		if curtime-close_time_eyes >= 5 and not left_eye_open:
			while not pressed("space"):
				if not playing:
					playing = True
					Thread(target=loopsound, args=(sound,), daemon=True).start()
				ret, frame = cap.read()
				cv2.imshow('Detection', cv2.flip(frame, 1))
				if cv2.waitKey(1) & 0xFF == ord('q'):
					running = False
					close()
			close_time_eyes = 0
			time_set_eyes = False
	else:
		time_set_eyes = False
	#Close when "q" is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		running = False
		close()