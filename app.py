from flask import Flask, request, jsonify
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import os
import math
from sklearn import neighbors
import pickle
app = Flask(__name__)


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image, model="cnn",
                                                                  number_of_times_to_upsample=0)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}"
                          .format(img_path,
                                  "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(x_face_locations, faces_encodings, knn_clf=None, model_path=None, distance_threshold=0.5):

    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations

    # If no faces are found in the image, return an empty results.
    if len(x_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    # faces_encodings = face_recognition.face_encodings(X_img_path, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]
    # print(closest_distances)

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings),
                                                                               x_face_locations, are_matches)]


@app.route('/train', methods=['GET'])
def train_data():
    train("./employee", model_save_path="test_model_2.clf")
    # print("Training complete!")
    return jsonify({'message': 'Success'})


@app.route('/process', methods=['POST'])
def process():
    frames = []
    frame_draw = []
    frame_count = 0
    file = request.json['path']
    full_path = './upload/'+str(file)
    if os.path.isfile(full_path):
        video_capture = cv2.VideoCapture(full_path)
        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        file_name = str(file).replace('.mp4', '')
        output_movie = cv2.VideoWriter('./results/'+str(file_name)+'.avi', fourcc, 15, (int(video_capture.get(3)),
                                                                           int(video_capture.get(4))))
        while output_movie.isOpened():

            ret, frame = video_capture.read()

            if not ret:
                break

            frames.append(frame)
            frame_draw.append(frame)
            frame_count = frame_count + 1

            # if frame_count == 100:
            #     break

            if len(frames) == 4:
                batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0,
                                                                                batch_size=4)

                # Now let's list all the faces we found in all 10 frames
                for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                    number_of_faces_in_frame = len(face_locations)

                    frame_number = frame_count - 4 + frame_number_in_batch
                    print(frame_number_in_batch)
                    print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))
                    if face_locations:
                        frame_draw_img = frame_draw[frame_number]
                        face_encodings = face_recognition.face_encodings(frame_draw_img, face_locations,
                                                                         num_jitters=10)
                        predictions = predict(face_locations, face_encodings, model_path="test_model_2.clf")
                        # print(predictions)
                        for name, (top, right, bottom, left) in predictions:

                            if name == "unknown":
                                # Draw rectangle
                                # print(
                                #     " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, "
                                #     "Right: {}".format(
                                #         top, left, bottom, right))

                                # Draw a label with a name below the face
                                #cv2.rectangle(frame_draw_img, (left, top), (right, bottom), (0, 255, 0), 2)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                #cv2.putText(frame_draw_img, "Not Black List", (left + 6, bottom - 6),
                                #            font, 0.5, (255, 255, 255), 1)
                            else:
                                # Draw rectangle
                                # print(
                                #     " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, "
                                #     "Right: {}".format(
                                #         top, left, bottom, right))

                                # Draw a label with a name below the face
                                cv2.rectangle(frame_draw_img, (left, top), (right, bottom), (0, 0, 255), 3)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                # cv2.putText(frame_draw_img, 'Black List', (left + 6, bottom - 6),
                                #             font, 0.5, (255, 255, 255), 1)
                                cv2.putText(frame_draw_img, name, (left + 6, bottom - 6),
                                            font, 0.75, (255, 255, 255), 2)

                        output_movie.write(frame_draw_img)

                # Clear the frames array to start the next batch
                frames = []

        return jsonify({'path': 'results/' + str(file_name)+'.avi'})

    else:
        return jsonify({'path': None,
                        'status': False})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='4477')
