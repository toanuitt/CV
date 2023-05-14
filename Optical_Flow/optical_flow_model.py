import cv2
import numpy as np

# Function to perform Single Object Tracking using optical flow
def track_object(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error opening video file")
        return

    # Select the initial bounding box around the object
    bbox = cv2.selectROI(frame, False)

    # Convert bounding box coordinates to integers
    bbox = tuple(map(int, bbox))

    # Initialize the previous frame and the points for optical flow
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)

    # Iterate through the frames
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None)

        # Select good points
        
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = prev_pts[status == 1]

        # Estimate the motion vector
            motion_vector = np.mean(good_new - good_old, axis=0)

        # Update the bounding box position based on the motion vector
            x, y, w, h = bbox
            motion_vector = motion_vector.astype(int)
            bbox = (int(x + motion_vector[0]), int(y + motion_vector[1]), w, h)
        else:
            pass

        # Draw bounding box
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        # Update the previous frame and points for the next iteration
        prev_frame = curr_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # Display the frame
        cv2.imshow('Single Object Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the Single Object Tracking function
track_object('run.mp4')