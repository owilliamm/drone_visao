from qreader import QReader
from threading import Thread, Lock
from pyzbar.pyzbar import decode
import numpy as np
import cv2
import queue

url = "http://192.168.1.199:8889/video"
cap = cv2.VideoCapture(url)
qreader = QReader(model_size='s')

q = queue.Queue(maxsize=2)
lock_qr = Lock()
lock_cdb = Lock()

resultados_qr = ([], [])
resultados_cdb = []
ultimo_qr = None
ultimo_cdb = None

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
            with lock_qr:
                resultados_qr = resultado

def processar_cdb():
    global resultados_cdb
    while True:
        if not q.empty():
            frame_scan = q.queue[-1].copy()
            cinza = cv2.cvtColor(frame_scan, cv2.COLOR_BGR2GRAY)
            decodificar = decode(cinza)
            
            resultados_temp = []
            for barcode in decodificar:
                texto = barcode.data.decode('utf_8')
                pts = [(p.x, p.y) for p in barcode.polygon]
                resultados_temp.append({
                    "texto": texto,
                    "bbox": pts,
                })
            with lock_cdb:
                resultados_cdb = resultados_temp

Thread(target=pegarframe, daemon=True).start()
Thread(target=processar_qr, daemon=True).start()
Thread(target=processar_cdb, daemon=True).start()

while True:
    frame = q.get()

    with lock_qr:
        textos, caixas = resultados_qr

    if not textos: ultimo_qr = None
    
    for i, caixa in enumerate(caixas):
        bbox = caixa.get('bbox_xyxy')
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)  
            infoqr = textos[i]
            if infoqr and infoqr != ultimo_qr: # ok, isso foi só pra não ficar enchendo o terminal
                print(f"Conteudo QR: {infoqr}")
                ultimo_qr = infoqr
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)

    with lock_cdb:
        lista_cdb = resultados_cdb[:]

    for item in lista_cdb:
        if item["bbox"] is not None:
            pts = np.array(item["bbox"], np.int32).reshape(-1, 1, 2)
            infocdb = item["texto"]
            if infocdb and infocdb != ultimo_cdb:
                print(f"Conteudo CDB: {infocdb}")
                ultimo_cdb = infocdb

                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

    cv2.imshow("Visao do Drone", frame)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Visao do Drone", cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
cv2.destroyAllWindows()