import cv2
from ultralytics import YOLO
import numpy as np

direccion_video = 'corto_futbol.mp4' 
escritura_direccion_video = 'corto_futbol_seguimiento.mp4' 
nombre_modelo = 'yolov8n.pt' 

Cantidad_defensores = 4 
cuadro_ancho_limite = 0.5 

# Colores para visualización
color_circulo = (0, 255, 0)  
color_linea = (0, 255, 0)    
color_caja = (255, 0, 0)    
color_texto = (255, 255, 255) 

model = YOLO(nombre_modelo)

# Lectura de video
cap = cv2.VideoCapture(direccion_video)
cuadro_ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cuadro_altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) # Get FPS for accurate output video

# Formato de vídeo de salida

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
out = cv2.VideoWriter(escritura_direccion_video, fourcc, fps, (cuadro_ancho, cuadro_altura))

# Bucle de procesamiento
cuadro_count = 0
while True: # Loop until break
    success, cuadro = cap.read()
    if not success:
        print("Fin del vídeo.")
        break 

    cuadro_count += 1

    # Corre seguimiento

    results = model(cuadro, verbose = True)
    #results = model(cuadro, verbose = True, classes = [2]) # A usar con el modelo de detección de futbol

    personas_en_imagen = [] # Para guardar {track_id: [cx, cy, x1, y1, x2, y2]}

    cajas = results[0].boxes.xyxy.cpu().numpy().astype(int)
    #track_ids = results[0].boxes.id.cpu().numpy().astype(int)

    for caja in cajas:
        x1, y1, x2, y2 = caja
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        personas_en_imagen.append([cx, cy, x1, y1, x2, y2])

        # Dibuja el cuadro alrededor de los jugadores y el ID de cada uno.
        cv2.rectangle(cuadro, (x1, y1), (x2, y2), color_caja, 2)

    # Identificacion de la línea defensiva a la izquierda
    puntos_linea_defensa = []
    if personas_en_imagen:
        potenciales_defensores = []
        limite_x = cuadro_ancho * cuadro_ancho_limite

        # 1. Selecciona solamente a los jugadores en la parte izquierda
        for data in personas_en_imagen:
            cx = data[0] # Use cx
            if cx < limite_x:
                potenciales_defensores.append(data)

        # 2. Selecciona los más cercanos a la izquierda
        if potenciales_defensores:
            # Ordena por cx
            sorted_potenciales_defensores = sorted(potenciales_defensores,
                                               key = lambda item: item[0], 
                                               reverse = False) 
            # Toma los 4(o N que nos interesan)
            candidatos_defensivos = sorted_potenciales_defensores[:Cantidad_defensores]

            # Extrae los puntos centrales para dibujar lo que queremos
            if candidatos_defensivos:
                puntos = [(data[0], data[1]) for data in candidatos_defensivos]

                # Ordenamos de arriba abajo, para dibujar la línea conectora
                puntos_linea_defensa = sorted(puntos, key=lambda p: p[1])

                # Dibuja círculos en los defensores
                for data in candidatos_defensivos:
                     cx, cy = data[0], data[1]

                     radius = max(5, int((data[4] - data[2]) * 0.3))
                     cv2.circle(cuadro, (cx, cy), radius, color_circulo, 2)

    # Dibujar la línea conectora
    if len(puntos_linea_defensa) >= 2:
        for i in range(len(puntos_linea_defensa) - 1):
            p1 = puntos_linea_defensa[i]
            p2 = puntos_linea_defensa[i + 1]
            cv2.line(cuadro, p1, p2, color_linea, 3)
    # Guardar cada cuadro
    out.write(cuadro)

cap.release()
out.release() 
print(f"Procesamiento listo. Video guardado en {escritura_direccion_video}")