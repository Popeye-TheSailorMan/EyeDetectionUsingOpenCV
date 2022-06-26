import cv2

camera = cv2.VideoCapture(1)
classifier = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_feat = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
lips_feat = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/mouth.xml')
nose_feat = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/nose.xml')

while(True):
    ret, img = camera.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_feat.detectMultiScale(roi_gray,minNeighbors=15)
        lips = lips_feat.detectMultiScale(roi_gray,minNeighbors = 10)
        nose = nose_feat.detectMultiScale(roi_gray,minNeighbors = 15)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # for (lx,ly,lw,lh) in lips:
        #     cv2.rectangle(roi_color, (lx, ly), (lx + lw, ly + lh), (0, 255, 0),2)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
camera.release()
cv2.destroyAllWindows()

