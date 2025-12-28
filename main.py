import cv2
from ultralytics import YOLO

print("Cargando modelo YOLO")
model = YOLO('yolov8s-world.pt') 

vocabulario_aula_moderna = [
    "person", "professor", "student",
    "laptop", "computer mouse", "keyboard", "computer monitor", "screen",
    "projector", "projector screen",
    "cell phone", "smartphone", "tablet", "headphones",
    "whiteboard", "marker", "pen", "pencil", "notebook", "backpack",
    "desk", "chair", "trash can", "clock", "glasses", "water bottle"
]

traducciones = {
    "person": "persona", "professor": "profesor", "student": "estudiante",
    "laptop": "laptop", "computer mouse": "mouse", "keyboard": "teclado",
    "computer monitor": "monitor", "screen": "pantalla",
    "projector": "proyector", "projector screen": "pantalla proyector",
    "cell phone": "celular", "smartphone": "celular", "tablet": "tablet",
    "headphones": "audífonos",
    "whiteboard": "pizarra", "marker": "marcador", "pen": "bolígrafo",
    "pencil": "lápiz", "notebook": "cuaderno", "backpack": "mochila",
    "desk": "escritorio", "chair": "silla", "trash can": "basurero",
    "clock": "reloj", "glasses": "lentes", "water bottle": "botella de agua"
}

model.set_classes(vocabulario_aula_moderna)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("no se detecta la camara")
    exit()

print("Iniciado...")
print("presiona q para salir")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(img, conf=0.30, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cls_id = int(box.cls[0])
            nombre_ingles = model.names[cls_id]
            nombre_espanol = traducciones.get(nombre_ingles, nombre_ingles)
            
            conf = float(box.conf[0])

            color_caja = (0, 255, 127) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color_caja, 2)
            
            label = f"{nombre_espanol} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), (0, 0, 0), -1)
            
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)

    cv2.imshow('Ojo Virtual - Dilan', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()