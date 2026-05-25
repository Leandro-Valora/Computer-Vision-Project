# import modules
import cv2
import os

# open file
cartella = os.path.dirname(os.path.abspath(__file__))
cartella2 = os.path.join(cartella, '..')
c3 = os.path.join(cartella2, '/video/')
pathVideo = os.path.join(cartella, 'video.mp4')

print(cartella)
print(cartella2)
print(c3)
print(pathVideo)

cap = cv2.VideoCapture(pathVideo)
print(f"Path usato: {pathVideo}")

if not cap.isOpened():
    print(f"ERRORE: video non aperto. Path usato:\n  '{cap}'")
    exit()
else:
    print(f"OK: video aperto correttamente:\n  '{cap}'")


# get FPS of input video
fps = cap.get(cv2.CAP_PROP_FPS)

# define output video and it's FPS
output_file = 'output.mp4'
output_fps = fps * 2

# define VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps,
                      (int(cap.get(3)), int(cap.get(4))))

# read and write frams for output video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("not ret!")
        break

    out.write(frame)

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# download output video on local machine
#files.download(output_file)    

#-------------------------------------------------------- nuovo codice --------------------------------------------------

import cv2 as cv
import sys

# ==========================================
# 1. DATABASE DEI VIDEO E GESTIONE SCELTA
# ==========================================

# Lista contenente i titoli dei video disponibili nella cartella
videosName = [
    "American Pilot Skillfully Dodges Incoming Missile Strike", 
    "AT4 Rocket, Javelin Missile & TOW Missile Live-fire",
    "Drone Chases Russian Soldier Back to His Unit’s Position", 
    "Fighter Jet Launches Missile Mid-Air - Epic Moment Caught!", 
    "Night Vision Ukraine Drone Chasing Russian Soldiers", 
    "Small Tornado on a Parking Lot", 
    "Static-Crossroads Traffic Aerial View",
    "TOW Missile vs T-72 Tank In Slow Motion", 
    "Traffic Shockwaves - Real Life Traffic Wave - Phantom Jam Timelapse", 
    "Ukraine war - Russian Su-25 Shot Down Its Own Wingman #warinukraine"
]

# Funzione che simula uno "switch-case" per associare il numero inserito all'indice dell'array
def switchVideo(videoElement):
    # Convertiamo l'input in intero per evitare una lunga catena di if/elif con stringhe
    try:
        scelta = int(videoElement) - 1
        # Controlliamo che l'indice sia valido (compreso tra 0 e la lunghezza della lista)
        if 0 <= scelta < len(videosName):
            return videosName[scelta]
    except ValueError:
        pass
    return None

# Mostriamo il menu grafico sulla console
print("\n ----------------------------------")
print(" ---   Menu Video Disponibili   ---")
print(" ---------------------------------- \n")

for i, v in enumerate(videosName):
    print(f"  {i+1}. {v}")

# Input utente per la scelta del video
selectVideo = input("\nQuale video vuoi analizzare? ")
menuChose = switchVideo(selectVideo)

# Se l'utente inserisce un numero non valido, il programma si ferma subito
if menuChose is None:
    print("Scelta non valida! Uscita in corso...")
    sys.exit()

# ==========================================
# 2. SELEZIONE DELL'ALGORITMO
# ==========================================
while True:
    algChose = input("\n Scegli Algoritmo:\n  1. MOG\n  2. MOG2\n ")
    if algChose in ("1", "2"):
        break
    print("  Inserisci 1 o 2")

# ==========================================
# 3. INIZIALIZZAZIONE VIDEO E SOTTRATTORE SFONDO
# ==========================================

# Costruiamo il percorso assoluto del file video .mp4
pathVideo = 'C:/Users/A/Desktop/MAGISTRALE/Fondamenti IA/Progetto Esame/video/' + menuChose + '.mp4'
video = cv.VideoCapture(pathVideo)

# Controllo di sicurezza: verifichiamo se OpenCV riesce ad accedere al file video
if not video.isOpened():
    print(f"ERRORE: Impossibile aprire il video. Controlla il percorso:\n  '{pathVideo}'")
    sys.exit()
else:
    print(f"\nOK: Video aperto correttamente:\n  '{pathVideo}'\n")

# Recuperiamo i fotogrammi per secondo (FPS) del video per calcolare il delay corretto tra i frame
fps = video.get(cv.CAP_PROP_FPS)
# Se il calcolo degli FPS fallisce o restituisce 0, impostiamo un valore di fallback a 30 FPS
if fps == 0: 
    fps = 30
delay = int(1000 / fps)

# [CORREZIONE FONDAMENTALE]: Inizializziamo il Background Subtractor UNA SOLA VOLTA FUORI dal ciclo while.
# In questo modo l'algoritmo può accumulare la storia dei frame precedenti per capire cosa è fermo e cosa si muove.
if algChose == "1":
    try:
        backSub = cv.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=10, backgroundRatio=0.7)
    except AttributeError:
        print("Errore: MOG1 richiede il pacchetto 'opencv-contrib-python'. Uso MOG2 di default.")
        backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
else:
    # MOG2 - Algoritmo più avanzato che gestisce meglio i cambi di luce e rileva le ombre (assegnando loro il valore grigio 127)
    backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Definiamo un elemento strutturante ellittico 5x5 per le operazioni morfologiche successive (pulizia del rumore)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# ==========================================
# 4. CICLO DI ELABORAZIONE FRAME BY FRAME
# ==========================================
while True:
    # Leggiamo il frame successivo del video
    ret, frame = video.read()
    
    # Se 'ret' è False, significa che il video è terminato o c'è stato un problema di lettura
    if not ret:
        print("Video terminato o nessun frame letto correttamente.")
        break

    # --- FASE DI PRE-PROCESSING ---
    # Ridimensionamento proporzionale: impostiamo il lato più lungo a 480 pixel per alleggerire il calcolo
    h, w = frame.shape[:2]
    scala = 480 / max(h, w)          
    nuovaW = int(w * scala)
    nuovaH = int(h * scala)
    frame_r = cv.resize(frame, (nuovaW, nuovaH))

    # Convertiamo in scala di grigi perché l'analisi del movimento non necessita delle informazioni sul colore
    gray = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
    
    # Applichiamo un filtro Sfocatura Gaussiana per eliminare il rumore ad alta frequenza (es. pixel che sfarfallano)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # --- APPLICAZIONE SOTTRAZIONE SFONDO ---
    # Applichiamo il modello dello sfondo aggiornato al frame corrente per ottenere la maschera di movimento
    mask = backSub.apply(blurred)

    # --- GESTIONE OMBRE E SOGLIATURA ---
    # MOG2 identifica le ombre e le colora di grigio (valore 127). 
    # Con questa sogliatura (threshold), teniamo solo i pixel sopra i 200 (bianco puro, movimento reale) 
    # e portiamo a 0 (nero) tutto il resto, eliminando efficacemente le ombre dal tracking.
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    # --- FASE DI POST-PROCESSING (Morfologia Matematica) ---
    # MORPH_OPEN: Rimuove i piccoli puntini bianchi isolati di rumore (erosione seguita da dilatazione)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    # MORPH_CLOSE: Riempie i piccoli "buchi" neri rimasti all'interno degli oggetti in movimento
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # --- FASE DI CONTOUR DETECTION (Rilevamento Contorni) ---
    # Troviamo i contorni geometrici delle aree bianche rimaste nella maschera pulita
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Calcoliamo l'area del contorno. Se è maggiore di 500 pixel, la consideriamo un oggetto rilevante (es. auto, persona)
        if cv.contourArea(cnt) > 500:  
            # Ricaviamo le coordinate del rettangolo che circoscrive il contorno
            x, y, w_box, h_box = cv.boundingRect(cnt)
            # Disegniamo il rettangolo verde sullo schermo (spessore 2 pixel) sul frame originale ridimensionato
            cv.rectangle(frame_r, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    # --- VISUALIZZAZIONE REALTÀ E CONTROLLI ---
    # Mostriamo a schermo il video originale con i rettangoli verdi e la maschera binaria del movimento
    cv.imshow("Frame", frame_r)
    cv.imshow("Mask",  mask)
    
    # Attendiamo il tempo calcolato (delay) basato sui FPS. Se l'utente preme 'q', interrompiamo il ciclo
    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

# Chiusura pulita delle risorse e delle finestre di OpenCV alla fine del programma
video.release()
cv.destroyAllWindows()
print("Programma terminato correttamente.")