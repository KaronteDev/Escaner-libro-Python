import cv2
print('OpenCV version:', cv2.__version__)
available = []
for i in range(6):
    print('Testing index', i)
    cap = None
    try:
        if hasattr(cv2, 'CAP_DSHOW'):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print('  isOpened: False')
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            print('  read failed')
            try:
                cap.release()
            except:
                pass
            continue
        print('  OK shape=', getattr(frame, 'shape', None), 'dtype=', getattr(frame, 'dtype', None), 'min=', frame.min(), 'max=', frame.max())
        available.append(i)
    except Exception as e:
        print('  error', e)
    finally:
        try:
            if cap is not None:
                cap.release()
        except:
            pass
print('Available indices:', available)
