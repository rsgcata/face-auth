# Arguments parser package
import argparse

# Async input output package
import asyncio

# JSON manipulation package
import json

# Logging package
import logging

# OS related functionality package (like dire/file paths constants)
import os
import glob

# SSL package
import ssl

# Unique unviersal IDs package
import uuid

# Import threading
import threading

import hashlib

# OpenCV - Computer vision package
import cv2

# HTTP input/ouput package
from aiohttp import web

# Audio/video package to work with the frames
from av import VideoFrame

# WebRTC package for real time communication (camera, video audio)
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

from userRepository import getUserByEmail, getUserById, storeUser

from faceTraining import trainUserData

import json

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
frameIndex = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the cascade classifier to be used for face detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Inherting the base class 'Thread' 
class AsyncWrite(threading.Thread):  
  
    def __init__(self, userId, frameIndex, grayscaleImage): 
  
        # calling superclass init 
        threading.Thread.__init__(self)
        self.userId = userId
        self.frameIndex = frameIndex
        self.grayscaleImage = grayscaleImage
  
    def run(self):
        cv2.imwrite("faceImages/" + self.userId + '/' +  
                    str(self.frameIndex) + ".jpg", 
                    self.grayscaleImage)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, peerConnection, transform, email):
        super().__init__()  # needed to call the parent construct
        self.track = track
        self.transform = transform
        self.email = email
        self.peerConnection = peerConnection

        self.smileAttemptsPassed = 0
        self.smileAttempts = 0
        self.smiledNumFrames = 0
        self.notSmiledNumFrames = 0
        self.livenessAttemptFrameCount = 0
        self.faceIdMatches = 0

        self.loggedInMessageSent = False

        emailHash = hashlib.sha1()
        emailHash.update(str(email).encode('utf-8'))

        self.emailHash = emailHash.hexdigest()
        self.user = storeUser(email)
        self.userId = str(self.user['id'])

        dirPath = os.path.dirname(os.path.realpath(__file__))

        # Create the dir where to save the face images
        userImagesPath = dirPath + os.path.sep + 'faceImages' + os.path.sep + self.userId
        if not os.path.exists(userImagesPath):
            os.mkdir(userImagesPath)

        if transform == 'login':
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read('trainedData' + os.path.sep + str(self.userId) + '.trainer.yml')

    # Async method required to wait for the frame sent by the client
    async def recv(self):
        global frameIndex, faceCascade
        frameIndex += 1
        frame = await self.track.recv()

        # Transform the frame so we can manipulate it with OpenCV
        img = frame.to_ndarray(format="bgr24")

        # Mutate the image colors to have only shades of gray, white to black range
        # Needed for face detection
        grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        # Adjusting the parameters can make the detection slower with better results
        # or faster with worse results
        faces = faceCascade.detectMultiScale(
            # Gray scaled image
            grayscaleImage,
            # The scale factor. How much the image should be scaled to match the trained
            # model. A lower value can have better results at matching faces but will be
            # slower. Image will be reduced in size by the given scale if the sliding
            # window did not detect any faces
            scaleFactor=1.05,
            # How many neighboring windows have detected faces. Higher value
            # may improve results of detection but can also miss some faces
            minNeighbors=10,
            # Minimum rectangle size in pixels to consider the window/rectangle a face
            # width / height
            minSize=(200, 200),
            # Flag if the image should be scaled
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Get the faces from the frame and return the frame with
        # a square around the detected face
        if self.transform == "createAccount":
            # Draw a rectangle around the faces and save the detected face as image
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                background = AsyncWrite(self.userId, frameIndex, grayscaleImage[y:y+h,x:x+w]) 
                background.start()
        else:
            # Get the confidence level for the user and add it to the frame  
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                smiles = smileCascade.detectMultiScale(
                    grayscaleImage[y:y+h,x:x+w],
                    scaleFactor=2,
                    minNeighbors=60
                )

                messageText = ""

                if self.smileAttempts >= 2:
                    # Total 360 frames at a frame rate of 30/s should be checked
                    # If both liveness steps are good and at least 300 frames matched the face
                    # then the user should be logged in
                    if self.smileAttemptsPassed == 2 and self.faceIdMatches > 300:
                        if not self.loggedInMessageSent:
                            jsonData = json.dumps({'isLoggedIn': True, 'email': self.email})
                            self.peerConnection.dataChannel.send(jsonData)
                            self.loggedInMessageSent = True
                        messageText = "Logged in"
                    else:
                        if not self.loggedInMessageSent:
                            jsonData = json.dumps({'isLoggedIn': False})
                            self.peerConnection.dataChannel.send(jsonData)
                            self.loggedInMessageSent = True
                        messageText = "Login failed"
                else:
                    if self.livenessAttemptFrameCount < 90:
                        messageText = "Smile"
                    else:
                        messageText = "Stop smiling"

                cv2.putText(
                    img, 
                    messageText, 
                    (x+5,y-25), 
                    font, 
                    0.5, 
                    (255,255,255), 
                    2
                )

                if self.livenessAttemptFrameCount < 90:
                    if len(smiles) > 0:
                        self.smiledNumFrames = self.smiledNumFrames + 1
                    self.livenessAttemptFrameCount = self.livenessAttemptFrameCount + 1
                elif self.livenessAttemptFrameCount < 180:
                    if len(smiles) == 0:
                        self.notSmiledNumFrames = self.notSmiledNumFrames + 1
                    self.livenessAttemptFrameCount = self.livenessAttemptFrameCount + 1
                else:
                    self.smileAttempts = self.smileAttempts + 1
                    smileRatio = self.smiledNumFrames * 100 / 90
                    notSmileratio = self.notSmiledNumFrames * 100 / 90

                    logger.info("Ratios " + str(smileRatio) + " " + str(notSmileratio))

                    if smileRatio > 55 and notSmileratio > 55:
                        self.smileAttemptsPassed = self.smileAttemptsPassed + 1
                    
                    self.smiledNumFrames = 0
                    self.notSmiledNumFrames = 0
                    self.livenessAttemptFrameCount = 0

                for (sx, sy, sw, sh) in smiles: 
                    cv2.rectangle(img, (sx + x, sy + y), ((sx + x + sw), (sy + y + sh)), (0, 0, 255), 2)

                id, confidence = self.recognizer.predict(grayscaleImage[y:y+h,x:x+w])

                if (confidence > 25):
                    id = "unknown"
                else:
                    self.faceIdMatches = self.faceIdMatches + 1
                    id = "ID: " + str(id) + " Email: " + self.user['email']
                
                confidence = "  {0}".format(round(confidence))

                cv2.putText(
                            img, 
                            str(id), 
                            (x+5,y-5), 
                            font, 
                            0.5, 
                            (255,255,255), 
                            2
                        )
                cv2.putText(
                            img, 
                            str(confidence), 
                            (x+5,y+h-5), 
                            font, 
                            0.5, 
                            (255,255,0), 
                            1
                        )

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        pc.dataChannel = channel

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            local_video = VideoTransformTrack(
                track, pc, transform=params["video_transform"], email=params["email"]
            )            

            # Remove old images for user to create a new set for training
            if local_video.transform == 'createAccount':
                files = glob.glob('faceImages/' + str(local_video.userId) + '/*')
                for f in files:
                    os.remove(f)

            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()
            if local_video.transform == 'createAccount':
                trainUserData(local_video.userId)
                log_info("Finished training the model for user: " + local_video.email)

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
