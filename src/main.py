try:
  import numpy as np
  import os, sys
  import cv2
  import keyboard
  import time
  import shutil
  import threading as THC
  from discord_webhook import DiscordWebhook, DiscordEmbed
  from helper import functions as funcs
except ModuleNotFoundError as e:
  print(f'ERROR: An error occurred when importing dependencies. \n{e}\n')
  sys.exit(1)


class globals:
  STD_DIMENSIONS = {
    '480p': (640, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4k': (3840, 2160),
  }
  frame_rate = 20.0  # Pretty CPU intensive '10.0' is good!
  resolution = (1280, 720)
  MODE = 'FACE-CAPTURE'  # Valid modes: 'FACE-CAPTURE' / 'MOTION-CAPTURE'
  last_message_sent = os.popen('time /t').read().replace('\n', '')  # Changes during runtime
  webhook_url = 'https://discord.com/api/webhooks/1140680728519127092/9_a9xGQKaE59MiyY8r8SkaA6RwceDBYXlFe6eW8NqEtr0hoBJzaNyqRfJ7_SDtIkrWz1'
  webhook = DiscordWebhook(url=webhook_url)


class LocalHelper:

  def GetURLSafeTime():
    out = os.popen('time /t').read().replace('\n', '')
    time = out.replace(':', '-').replace(' ', '')
    return time

  def sendDiscordAlert(*file_path):
    time = os.popen('time /t').read().replace('\n', '')
    if time == globals.last_message_sent:
      return 0  # Must wait at least a minute
    else:
      globals.last_message_sent = time
      embed = DiscordEmbed()
      globals.webhook.wait = True
      for file in file_path:
        file_name = file.split('/')[-1]

        with open(file, 'rb') as f:
          file_data = f.read()

        globals.webhook.add_file(file_data, file_name)

      embed.set_title('Scout Alert')
      embed.set_description(f'Scout detected a face: {funcs.getTime()}')
      globals.webhook.add_embed(embed)
      response = globals.webhook.api_post_request()
      print(response)

  def zipFiles(target_files, faceORvideo):
    if faceORvideo == 'facedump': ideal_ext = '.png'
    elif faceORvideo == 'video': ideal_ext = '.avi'
    else: return 0  # Pass the right arg bro...
    dump_dir = f'{faceORvideo}/dump #{funcs.uniqueIDGen()} {LocalHelper.GetURLSafeTime()}'
    target_files = list(target_files)
    os.mkdir(dump_dir)
    for target in target_files:
      target_name = target.split('/')[-1]
      target_ext = target_name[target_name.find('.'):]
      if target_ext == ideal_ext:
        with open(target, 'rb') as Fin:
          content = Fin.read()
        with open(f'{dump_dir}/{target_name}', 'wb') as Fout:
          Fout.write(content)
      else:
        target_files.remove(target)

    zip_file = shutil.make_archive(dump_dir, 'zip', dump_dir)
    shutil.rmtree(dump_dir)
    for target in target_files:
      if target.endswith(ideal_ext):
        try: os.remove(target)
        except: pass  # Tried to delete working video


# TODO's
'''
* OPTIMIZE
* Implement a snapshot limiter like one every 30 secs
* Maybe lower minSize to increase accuracy
* Make it send a 'F15' input so we dont go to sleep
* Add a movement detector maybe??
* Implement threading to help with consistency and pauses
* Implement eye detection in test.py ✔️
* Create a cap on amount of snapshots (Compressed them instead) ✔️
* Snapshot every time a face is detected and save it ✔️
* Output video file ✔️
'''


class faceDetection:

  def __init__(self) -> None:
    video_output_name = f'video/cam #{funcs.uniqueIDGen()} {LocalHelper.GetURLSafeTime()}.avi'
    faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_name, fourcc, globals.frame_rate, globals.resolution)
    cap = cv2.VideoCapture(0)
    cap.set(3, globals.resolution[0])  # set Width
    cap.set(4, globals.resolution[1])  # set Height

    while True:
      ret, img = cap.read()
      img = cv2.flip(img, 1)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
      )

      for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

      out.write(img)
      cv2.imshow('video', img)

      if str(faces) == '()':
        pass
      else:
        print(f'Face found: {funcs.getTime()}')
        snapshot_file = f'facedump/snapshot #{funcs.uniqueIDGen()}.png'
        cv2.imwrite(snapshot_file, img)
        LocalHelper.sendDiscordAlert(video_output_name, snapshot_file)

      if len(os.listdir('facedump')) >= 50:
        def dumpCheck():
          print('OVERFLOW')
          files = []
          for file in os.listdir('facedump'):
            file = f'facedump/{file}'
            files.append(file)
          LocalHelper.zipFiles(files, 'facedump')

        thread1 = THC.Thread(dumpCheck())
        thread1.start()
      if len(os.listdir('video')) >= 35:
        def videoCheck():
          print('OVERFLOW')
          files = []
          for file in os.listdir('video'):
            file = f'video/{file}'
            files.append(file)
          LocalHelper.zipFiles(files, 'video')

        thread2 = THC.Thread(videoCheck())
        thread2.start()

      key = cv2.waitKey(30) & 0xff
      if key == 27:  # press 'ESC' to quit
        break
      else:
        time.sleep(0.2)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


class faceEyeDetection:  # This shit is janky at best...

  def __init__(self) -> None:
    video_output_name = f'video/cam #{funcs.uniqueIDGen()} {LocalHelper.GetURLSafeTime()}.avi'
    faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_name, fourcc, globals.frame_rate, globals.resolution)
    cap = cv2.VideoCapture(0)
    cap.set(3, globals.resolution[0])  # set Width
    cap.set(4, globals.resolution[1])  # set Height

    while True:
      ret, img = cap.read()
      img = cv2.flip(img, 1)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
      )

      for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
          roi_gray,
          scaleFactor=1.4,
          minNeighbors=10,
          minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
          cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

      out.write(img)
      cv2.imshow('video', img)

      if str(faces) == '()' or str(eyes) == '()':  # Switching 'and' to 'or' fixes it
        pass
      else:
        if str(faces) != '()':
          print(f'Face found: {funcs.getTime()}')
        if str(eyes) != '()':
          print(f'Eye found: {funcs.getTime()}')
        snapshot_file = f'facedump/snapshot #{funcs.uniqueIDGen()}.png'
        cv2.imwrite(snapshot_file, img)
        LocalHelper.sendDiscordAlert(video_output_name, snapshot_file)

      if len(os.listdir('facedump')) >= 50:
        def dumpCheck():
          print('IMAGE OVERFLOW')
          filesA = []
          for fileA in os.listdir('facedump'):
            fileA = f'facedump/{fileA}'
            filesA.append(fileA)
          LocalHelper.zipFiles(filesA, 'facedump')

        thread1 = THC.Thread(dumpCheck())
        thread1.start()
      if len(os.listdir('video')) >= 35:
        def videoCheck():
          print('VIDEO OVERFLOW')
          filesB = []
          for fileB in os.listdir('video'):
            fileB = f'video/{fileB}'
            filesB.append(fileB)
          LocalHelper.zipFiles(filesB, 'video')

        thread2 = THC.Thread(videoCheck())
        thread2.start()

      key = cv2.waitKey(30) & 0xff
      if key == 27:  # press 'ESC' to quit
        break
      else:
        time.sleep(0.2)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


faceEyeDetection()
