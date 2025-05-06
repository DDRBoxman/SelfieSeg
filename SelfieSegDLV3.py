from PIL import Image
import cv2
import time
import numpy as np
from torchvision import models
import torch
from torchvision import transforms

class SelfieSegDLV3:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.dev = "cpu"    #"cuda"
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()

    def seg(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Comment the Resize and CenterCrop for better inference results
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        colors = np.array([(0, 0, 0),  # 0=background
             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
             (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
             (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
             (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 255),
             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
             (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

        # plot the semantic segmentation predictions of 21 classes in each color
        out = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(img.size)
        out.putpalette(colors)
        out = np.array(out)

        mask = (255 * (out / 15)).astype("uint8")  # 15=person
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask

if __name__ == "__main__":
    #"""
    width = 480
    height = 360
    seg = SelfieSegDLV3(width, height)

    # Capture video from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Load and resize the background image
    bgd = cv2.imread('./images/background.jpeg')
    if bgd is None:
        raise FileNotFoundError("Could not load background image from './images/background.jpeg'")
    
    # Get actual frame dimensions after camera setup
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from camera")
    actual_height, actual_width = test_frame.shape[:2]
    
    # Resize background to match actual frame dimensions
    bgd = cv2.resize(bgd, (actual_width, actual_height))

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
        
        # Debug prints
        print(f"Frame shape: {frame.shape}")
        print(f"Background shape: {bgd.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # Ensure mask has same dimensions as frame
        mask = cv2.resize(mask, (actual_width, actual_height))
        
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
