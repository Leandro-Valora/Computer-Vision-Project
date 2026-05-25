import cv2 as cv
import sys
import os

# ==========================================
# 1. DATABASE DEI VIDEO E GESTIONE SCELTA
# ==========================================

# Array contenente i titoli dei video nella cartella
videosName = [
    "American Pilot Skillfully Dodges Incoming Missile Strike", "AT4 Rocket, Javelin Missile & TOW Missile Live-fire",
    "Drone Chases Russian Soldier Back to His Unit’s Position", "Fighter Jet Launches Missile Mid-Air - Epic Moment Caught!", 
    "Night Vision Ukraine Drone Chasing Russian Soldiers", "Small Tornado on a Parking Lot", "Static-Crossroads Traffic Aerial View",
    "TOW Missile vs T-72 Tank In Slow Motion", "Traffic Shockwaves - Real Life Traffic Wave - Phantom Jam Timelapse", 
    "Ukraine war - Russian Su-25 Shot Down Its Own Wingman #warinukraine", "Pedestrian_vs_Car_Crash", 
    "Dash cam footage shows driver ignoring barricades"
    ]

# Switch case
def switchVideo(videoElement):
    if videoElement == "1":
        return videosName[0]
    elif videoElement == "2":
        return videosName[1]
    elif videoElement == "3":
        return videosName[2]
    elif videoElement == "4":
        return videosName[3]
    elif videoElement == "5":
        return videosName[4]
    elif videoElement == "6":
        return videosName[5]
    elif videoElement == "7":
        return videosName[6]
    elif videoElement == "8":
        return videosName[7]
    elif videoElement == "9":
        return videosName[8]
    elif videoElement == "10":
        return videosName[9]
    # elif videoElement == "11":
    #     return videosName[10]
    # elif videoElement == "12":
    #     return videosName[11]


# Menu
print("\n ----------------------------------")
print(" ---   Menu Video Disponibili   ---")
print(" ---------------------------------- \n")

for i, v in enumerate(videosName):
    print(f"  {i+1}. {v}")

selectVideo = input("\nQuale video vuoi analizzare ?  ")
menuChose = switchVideo(selectVideo)

# Check numero non valido
if menuChose is None:
    print("Scelta non valida! Uscita in corso...")
    sys.exit()


# ==========================================
# 2. SELEZIONE DELL'ALGORITMO
# ==========================================

while True:
    algChose = input("\n Scegli Algoritmo:\n  1. MOG\n  2. MOG2\n Scelta: ")
    if algChose in ("1", "2"):
        break
    print("  Inserisci 1 o 2 --> ")


# ======================================================
# 3. INIZIALIZZAZIONE VIDEO E SOTTRAZIONE DELLO SFONDO
# ======================================================

# Cattura dei frame del video
pathVideo = 'C:/Users/A/Desktop/MAGISTRALE/Fondamenti IA/Progetto Esame/video/' + menuChose + '.mp4'
video = cv.VideoCapture(pathVideo)

# Check file se è stato trovato
if not video.isOpened():
    print(f"ERRORE: video non aperto. Path usato:\n  '{video}'")
    exit()
else:
    print(f"Video aperto correttamente:\n  '{pathVideo}'")


# calcola il delay dal FPS del video
fps = video.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)


# =======================
# 3.1 DOWNLOAD MASK 
# =======================

downld = input("Vuoi scaricare il video (s o n) ? ")

if(downld is "s"):
    # Leggo le dimensioni reali del primo frame per inizializzare il writer
    ret_test, frame_test = video.read()
    if not ret_test:
        print("Errore: impossibile leggere il primo frame.")
        sys.exit()

    h0, w0 = frame_test.shape[:2]
    scala0  = 480 / max(h0, w0)
    outW    = int(w0 * scala0)
    outH    = int(h0 * scala0)

    # Riporto il video all'inizio dopo il frame di test
    video.set(cv.CAP_PROP_POS_FRAMES, 0)

    # ---- Cartella output ----
    cartellaOutput = 'C:/Users/A/Desktop/MAGISTRALE/Fondamenti IA/Progetto Esame/output/'
    os.makedirs(cartellaOutput, exist_ok=True)   # la crea se non esiste

    # Nome file output: es. "Small Tornado on a Parking Lot_MOG2_mask.mp4"
    nomeOutput = f"{menuChose}_{algChose}_mask.mp4"
    pathOutput = os.path.join(cartellaOutput, nomeOutput)

    # VideoWriter: codec mp4v, stessi FPS del video originale, dimensione frame ridimensionato
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(pathOutput, fourcc, fps, (outW, outH) )

    print(f"Output mask salvato in:\n  '{pathOutput}'")

# Inizializzo il Background Subtractor
if algChose == "1":
    try:
        backSub = cv.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=10, backgroundRatio=0.7)
    except AttributeError:
        print("Errore: MOG1 richiede il pacchetto 'opencv-contrib-python'. Uso MOG2 di default.")
else:
    # MOG2 - Algoritmo più avanzato che gestisce meglio i cambi di luce e rileva le ombre (assegnando loro il valore grigio 127)
    backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)


# Definiamo un elemento strutturante ellittico 5x5 per le operazioni morfologiche successive (pulizia del rumore)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


# ==========================================
# 4. CICLO DI ELABORAZIONE FRAME BY FRAME
# ==========================================

while True:
    ret, frame = video.read()
    if not ret:
        print("Video terminato o nessun frame letto correttamente!")
        break


    # --- FASE di PRE-PROCESSING ----

    h, w = frame.shape[:2]
    scala = 480 / max(h, w)          # scala rispetto al lato più lungo
    nuovaW = int(w * scala)
    nuovaH = int(h * scala)
    frame_r = cv.resize(frame, (nuovaW, nuovaH))
    #fisso --> frame_r = cv.resize(frame, (640, 360))

    # Convertiamo in scala di grigi
    gray = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
    
    # Applichiamo un filtro Sfocatura eliminando rumore
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # --- MOG ---
    mask = backSub.apply(blurred)
    
    # rimuove le ombre (127 = grigio delle ombre in MOG2)
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    # --- FASE di POST-PROCESSING ---
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # --- FASE di CONTOUR DETECTION ---
    # Troviamo i contorni geometrici delle aree bianche rimaste nella maschera pulita
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    oggettiRilevati = 0
    
    for cnt in contours:
        # Calcoliamo l'area del contorno. Se è maggiore di 500 pixel, la consideriamo un oggetto rilevante (es. auto, persona)
        if cv.contourArea(cnt) > 500:
            # Counter oggetti rilevati
            oggettiRilevati += 1  
            # Ricaviamo le coordinate del rettangolo che circoscrive il contorno
            x, y, w_box, h_box = cv.boundingRect(cnt)
            # Disegniamo il rettangolo verde sullo schermo (spessore 2 pixel) sul frame originale ridimensionato
            cv.rectangle(frame_r, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    cv.putText(

        frame_r,                          # immagine su cui scrivere
        f"Oggetti: {oggettiRilevati}",    # testo
        (10, 30),                         # posizione (x, y) in pixel
        cv.FONT_HERSHEY_SIMPLEX,          # font
        1.0,                              # scala font
        (0, 0, 255),                      # colore
        1                                 # spessore
    )

    if(downld is "s"):
        # --- SALVATAGGIO MASK A COLORI (il writer vuole BGR, non grayscale) ---
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        writer.write(mask_bgr)

    # --- VISUALIZZAZIONE REALTÀ E CONTROLLI ---
    # Mostriamo a schermo il video originale con i rettangoli verdi e la maschera binaria del movimento
    cv.imshow("Frame", frame_r)
    cv.imshow("Mask",  mask)
    
    # Attendiamo il tempo calcolato (delay) basato sui FPS. Se l'utente preme 'q', interrompiamo il ciclo
    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

# Chiusura pulita delle risorse e delle finestre di OpenCV alla fine del programma
video.release()
if downld is "s": writer.release()
cv.destroyAllWindows()
print("Programma terminato correttamente.")