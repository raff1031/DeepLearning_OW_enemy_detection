# OW-Detector — Pipeline Completa

## Panoramica

```
SCHERMO  →  CAPTURE  →  YOLO  →  OVERLAY
                ↑
          [TRAINING]
          DATASET → SPLIT → AUGMENT → FINETUNE
```

Il sistema ha due rami paralleli:
- **Ramo inferenza** (runtime): cattura → detect → disegna
- **Ramo training** (offline): raccoglie dati → allena → aggiorna pesi

---

## Step 1 — Screen Capture (`utils/capture.py`)

**Cosa fa:** cattura il frame corrente dello schermo usando `mss` (DirectX/DXGI su Windows).

**Come funziona:**
- `mss` legge direttamente il framebuffer GPU → latenza < 5ms
- Restituisce un array NumPy BGR (formato OpenCV)
- Supporta monitor multipli, regioni parziali

**Vantaggi:**
- Velocissimo (< 5ms per frame)
- Nessun hook nel gioco, zero rischio ban
- Funziona con qualsiasi gioco/applicazione

**Limiti:**
- Richiede che il gioco sia visibile su schermo (non minimizzato)
- In fullscreen esclusivo può fallire → usare borderless window

---

## Step 2 — Raccolta Dati (3 metodi)

### 2a — HSV Masking (`capture_loop.py`)
Filtra i pixel per colore dell'outline nemici (rosso in OW2).

```
frame → converti in HSV → maschera colore → trova contorni → salva box
```

**Vantaggi:** CPU only, zero dipendenze ML, velocissimo  
**Limiti:** dipende dal colore impostato in Daltonismo; rosso si spezza in H=0-10 + H=165-180; cattura falsi positivi (UI, effetti)

### 2b — Pseudo-labeling YOLO (`pseudolabel.py`)
Usa YOLOv8n pre-trainato su COCO per rilevare "person" (classe 0).

```
frame → YOLO COCO → box "person" → filtra per size/ratio → salva label
```

**Vantaggi:** più robusto dell'HSV; nessuna calibrazione colore  
**Limiti:** confonde teammate con nemici; confidence bassa su eroi in posa non umana (Orisa, Winston)

### 2c — Video YouTube (`video_raw.mp4`)
Scarica gameplay da YouTube, estrae 1 frame ogni 2 secondi, pseudo-labela con YOLO già trainato.

```
yt-dlp → mp4 → OpenCV frame extraction → YOLO → label
```

**Vantaggi:** centinaia di frame diversi in pochi minuti; scene mai viste dal modello  
**Limiti:** qualità variabile (360p-720p); il modello etichetta solo ciò che riconosce → label incomplete se il modello è debole

---

## Step 3 — Dataset Preparation (`data/dataset.py`)

Prende le immagini raw e le prepara per il training.

**Operazioni:**
1. **Split** 80/10/10 → `train / val / test`
2. **Augmentation** su train (×4 campioni per immagine):
   - Flip orizzontale (con correzione coordinate label)
   - Brightness jitter (fattore 0.6–1.4)
   - Gaussian blur (kernel 3 o 5)
   - Gaussian noise (σ=8)
3. **Genera `overwatch.yaml`** con path assoluti per YOLO

**Vantaggi:** augmentation semplice ma efficace; riproducibile (seed fisso)  
**Limiti:** augmentation non include rotazioni/crop → manca variabilità geometrica; niente MixUp/Mosaic custom (YOLO li fa internamente)

**Output:**
```
datasets/
  images/train/   → ~7000+ immagini
  images/val/     → ~230 immagini
  images/test/    → ~230 immagini
  labels/...      → label YOLO corrispondenti
data/overwatch.yaml
```

---

## Step 4 — Training (`training/train.py`)

Transfer learning da YOLOv8s pre-trainato su COCO (80 classi, dataset enorme).

### Strategia 3 fasi

| Fase | Epoch | Layer congelati | Learning Rate | Obiettivo |
|------|-------|-----------------|---------------|-----------|
| 1 | 1-30 | backbone (10 layer) | 1e-3 | allena solo la detection head |
| 2 | 31-70 | primi 7 layer | 3e-4 | fine-tuning leggero backbone |
| 3 | 71-100 | nessuno | 5e-5 | adattamento completo al dominio |

**Perché 3 fasi?**  
Il backbone COCO ha già imparato feature visive potenti (bordi, texture, forme). Congelandolo inizialmente si evita il "catastrophic forgetting" — si allena prima la testa a riconoscere 1 classe invece di 80, poi si sblocca gradualmente.

**Architettura YOLOv8s:**
```
Input 640×640
  → Backbone CSPNet (estrae feature: bordi, forme, texture)
  → Neck FPN+PAN (fonde feature multi-scala: oggetti grandi e piccoli)
  → Head Detect (predice box + confidence per 3 scale)
Output: box [cx, cy, w, h] + conf per ogni detection
```

**Loss functions:**
- `box_loss` — errore posizione/dimensione box (CIoU)
- `cls_loss` — errore classificazione (BCE, 1 sola classe)
- `dfl_loss` — Distribution Focal Loss, affina i bordi a livello pixel

**Vantaggi:** transfer learning = convergenza rapida; YOLOv8s = buon balance velocità/accuratezza  
**Limiti:** dataset gallery ≠ gameplay → domain gap; 1 sola classe (enemy) non distingue eroe specifico

---

## Step 5 — Inferenza (`inference/detect.py`)

Il modello gira in un thread separato per non bloccare l'overlay.

```
Thread Detector:
  loop → capture frame → YOLO predict → metti risultato in Queue

Thread Overlay:
  loop → leggi Queue → disegna box → aggiorna finestra
```

**Parametri chiave:**
- `conf=0.4` — soglia confidence (abbassa per più detection, alza per meno falsi positivi)
- `imgsz=640` — dimensione input (aumenta a 1280 per nemici lontani, più lento)
- `device="0"` — GPU (usa `"cpu"` se necessario)

**Performance attuali:**
- Inferenza: ~1.7ms/frame → teoricamente 580 FPS
- Con capture + overhead: 20-30 FPS real-time
- VRAM: ~2.3GB

**Vantaggi:** thread separati = overlay fluido anche se YOLO è lento; Queue thread-safe  
**Limiti:** latenza totale ~50ms (capture + infer + draw); nessun tracking tra frame (ogni frame è indipendente)

---

## Step 6 — Overlay ESP (`inference/overlay.py`)

Finestra PyQt5 trasparente sempre in primo piano.

**Proprietà finestra:**
- `FramelessWindowHint` — nessun bordo
- `WA_TranslucentBackground` — sfondo trasparente
- `WindowStaysOnTopHint` — sempre sopra
- `WindowTransparentForInput` — i click passano al gioco

**Cosa disegna:**
- Box rossi attorno ai nemici rilevati
- Label "enemy + confidence" sopra ogni box
- FPS counter in alto a sinistra
- Stato ESP (ON/OFF) con F9

**Vantaggi:** completamente non invasivo; il gioco non sa che esiste  
**Limiti:** in fullscreen esclusivo non funziona (usare borderless); su monitor secondario richiede calibrazione manuale posizione

---

## Metriche del Modello Attuale

| Metrica | Valore | Significato |
|---------|--------|-------------|
| mAP@0.5 | **97.1%** | trova il 97% dei nemici con IoU ≥ 50% |
| mAP@0.5:0.95 | **71.4%** | precisione media su soglie IoU più severe |
| Precision | 95.2% | quando dice "nemico", ha ragione il 95% delle volte |
| Recall | 94.5% | trova il 94% dei nemici presenti |
| Velocità | **1.7ms** | ~580 FPS teorici |

---

## Dataset Attuale

| Sorgente | Immagini | Qualità label | Note |
|----------|----------|---------------|------|
| Roboflow OW2 | 1583 | ★★★★★ | annotate a mano, gallery/skin |
| Gameplay live (HSV) | ~160 | ★★☆☆☆ | label da colore outline, rumoroso |
| Video YouTube (3 video) | ~760 | ★★★☆☆ | pseudo-label YOLO, gameplay reale |
| **Totale raw** | **~2500** | | |
| **Totale con augmentation** | **~7400** | | |

---

## Limiti Principali e Come Risolverli

### 1. Domain Gap (gallery → gameplay)
**Problema:** il modello vede nemici in posa statica durante training, ma in gameplay sono in movimento, parzialmente occlusi, con motion blur.  
**Soluzione:** raccogliere più frame gameplay reali (sia da live capture che da video) e fare fine-tuning ciclico.

### 2. Nemici lontani non rilevati
**Problema:** nemici piccoli (< 20px) hanno confidence bassa.  
**Soluzione A:** abbassa `conf` a 0.25-0.3 in `inference/detect.py`  
**Soluzione B:** usa `imgsz=1280` nel training (più lento ma vede dettagli piccoli)

### 3. Teammate vs Enemy
**Problema:** il modello non distingue alleati da nemici (1 sola classe).  
**Soluzione:** aggiungere classe 1 = "ally" e raccogliere frame con annotazioni separate. Richiede dataset con label distinte.

### 4. Nessun tracking
**Problema:** ogni frame è indipendente → i box "saltano" tra frame.  
**Soluzione:** aggiungere ByteTrack o BoTSORT (integrato in Ultralytics) per tracking ID stabile tra frame.

---

## Comandi Rapidi

```bash
# Calibra colore outline
python utils/hsv_tuner.py rosso

# Cattura frame in partita (20 min)
python capture_loop.py --duration 1200

# Prepara dataset + augmentation
python main.py --mode dataset --augment

# Training completo (100 epoch)
python main.py --mode train --epochs 100

# Fine-tuning su modello esistente (30 epoch)
python main.py --mode train --model weights/ow_detector.pt --epochs 30

# Lancia overlay live
python main.py --mode overlay

# Scarica video YT + estrai frame
python -m yt_dlp URL -o datasets/video.mp4
# poi esegui estrazione frame manuale con OpenCV + YOLO
```
