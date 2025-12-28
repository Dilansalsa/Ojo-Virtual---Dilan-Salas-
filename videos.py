import cv2
from ultralytics import YOLO

model = YOLO('yolov8s-world.pt')

vocabulario_sala = [
    "person", "guitar", "vase", "sofa", "couch", "chair", "table",
    "coffee table", "tv", "television", "remote", "lamp", "plant",
    "book", "shelf", "picture frame", "cushion", "pillow", "rug",
    "fireplace", "window", "door", "dog", "cat"
]

traducciones = {
    "person": "persona", "guitar": "guitarra", "vase": "florero",
    "sofa": "sofa", "couch": "sofa", "chair": "silla",
    "table": "mesa", "coffee table": "mesa de centro",
    "tv": "television", "television": "televisión",
    "remote": "control remoto", "lamp": "lampara",
    "plant": "planta", "book": "libro", "shelf": "estanteria",
    "picture frame": "cuadro", "cushion": "cojin",
    "pillow": "almohada", "rug": "alfombra",
    "fireplace": "chimenea", "window": "ventana",
    "door": "puerta", "dog": "perro", "cat": "gato"
}

model.set_classes(vocabulario_sala)

nombre_video = "Video_Casa_2.mp4"
#videos para probar
#Video_Casa_1.mp4
#Video_Casa_2.mp4
#Video_Casa_3.mp4
cap = cv2.VideoCapture(nombre_video)

if not cap.isOpened():
    print(f"no se pudo abrir el video '{nombre_video}'")
    exit()

print(f"procesando video: {nombre_video}")

while True:
    success, img = cap.read()
    if not success:
        print("fin del video")
        break

    results = model.predict(img, conf=0.3, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_id = int(box.cls[0])
            nombre_ingles = model.names[cls_id]
            nombre_espanol = traducciones.get(nombre_ingles, nombre_ingles)

            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(img, (x1, y1-25), (x1 + len(nombre_espanol)*15, y1), (0, 0, 0), -1)
            texto_pantalla = f"{nombre_espanol} {conf:.1f}"
            cv2.putText(img, texto_pantalla, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('deteccion en salas - Dilan', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()