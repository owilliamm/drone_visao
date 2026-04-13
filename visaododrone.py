from qreader import QReader
from threading import Thread
import cv2
import queue

url = "http://192.168.1.199:8889/video"
cap = cv2.VideoCapture(url)
qreader = QReader(model_size='s')

q = queue.Queue(maxsize=2)

resultados_qr = ([], [])
ultimo_texto_lido = None

def pegarframe():
    while True:
        ret, frame = cap.read()
        if not ret: break
        if q.full():
            q.get_nowait()
        q.put(frame)

def processar_qr():
    global resultados_qr
    while True:
        if not q.empty():
            frame_scan = q.queue[-1].copy()
            resultado = qreader.detect_and_decode(image=frame_scan, return_detections=True)
            resultados_qr = resultado

Thread(target=pegarframe, daemon=True).start()
Thread(target=processar_qr, daemon=True).start()

while True:
    frame = q.get()
    textos, caixas = resultados_qr

    if not textos: ultimo_texto_lido = None

    for i, caixa in enumerate(caixas):
        bbox = caixa.get('bbox_xyxy')
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)  
            info = textos[i]

            if info: # ok, isso foi só pra não ficar enchendo o terminal
                if info != ultimo_texto_lido:
                    print(f"Conteudo QR: {info}")
                    ultimo_texto_lido = info
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 1)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)


    cv2.imshow("Visao do Drone", frame)
    cv2.waitKey(1)
    
    if cv2.getWindowProperty("Visao do Drone", cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
cv2.destroyAllWindows()