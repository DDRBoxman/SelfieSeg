import cv2, sys, time
import numpy as np
from keras.models import load_model
from PIL import Image

class SelfieSegMNV3:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.dim = 224
        self.model = load_model("models/mnv3_seg/munet_mnv3_wm05.h5")

    def seg(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        image = image.resize((self.dim, self.dim), Image.LANCZOS)
        img = np.float32(np.array(image) / 255.0)
        img = img[:, :, 0:3]

        # Reshape input and threshold output
        out = self.model.predict(img.reshape(1, self.dim, self.dim, 3))
        out = np.float32((out > 0.5)).reshape(self.dim, self.dim)
        mask = (255 * out).astype("uint8")

        # Resize mask to match the ORIGINAL frame dimensions
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask

if __name__ == "__main__":
    width = 640  # Match with actual camera output
    height = 480  # Match with actual camera output
    seg = SelfieSegMNV3(width, height)

    # Capture video from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Load and resize the background image
    bgd = cv2.imread('./images/background.jpeg')
    bgd = cv2.resize(bgd, (width, height))

    elapsedTime = 0
    count = 0

    while cv2.waitKey(1) < 0:
        t1 = time.time()

        # Read input frames
        success, frame = cap.read()
        if not success:
           cap.release()
           break

        # Get segmentation mask
        mask = seg.seg(frame)

        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask shape: {mask.shape}")
        print(f"Frame shape: {frame.shape}")

        # Ensure mask is the right type
        mask = mask.astype(np.uint8)

        # No need to resize here as we're already resizing in the seg() method

        # Merge with background
        fg = cv2.bitwise_or(frame, frame, mask=mask)
        bg = cv2.bitwise_or(bgd, bgd, mask=cv2.bitwise_not(mask))  # Use bitwise_not instead of ~
        out = cv2.bitwise_or(fg, bg)

        elapsedTime += (time.time() - t1)
        count += 1
        fps = "{:.1f} FPS".format(count / elapsedTime)

        # Show output in window
        cv2.putText(out, fps, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 255, 38), 1, cv2.LINE_AA)
        cv2.imshow('Selfie Segmentation', out)

    cv2.destroyAllWindows()
    cap.release()