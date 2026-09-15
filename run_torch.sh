#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features all,raw,doppler --dataset-sim dataset_v2 --arch cnn --dataset-real dataset_real_v2 --activities FALL,IDLE,RUN,STAND,WALK
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features all,raw,doppler --dataset-sim dataset_v2 --arch cnn --dataset-real dataset_real_v2 --activities IDLE,STAND,WALK
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features all,raw,doppler --dataset-sim dataset_v2 --arch cnn --dataset-real dataset_real_v2 --activities FALL,IDLE,JUMP,RUN,STAND,WALK #--subtract-mode ema --ema-alpha 0.80
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features all,raw,doppler --dataset-sim dataset_v2 --arch cnn,rf,svm --dataset-real dataset_real_v2 --activities FALL,IDLE,RUN,STAND,WALK #--subtract-mode ema --ema-alpha 0.80

###dataset con la griglia dei ricevitori  e il ricevitore del real è l'rx8 [4,-1.5,1.0] (quindi all'interno della griglia)
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features doppler  --dataset-sim dataset_vx3  --arch cnn,rf,svm --dataset-real dataset_real_v3  --activities FALL,IDLE,RUN,STAND,WALK,JUMP

###dataset con la griglia dei ricevitori  e il ricevitore del real è l'rx0 originale, e quindi all'esterno della griglia (out_of_grid)
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py   --features doppler  --dataset-sim dataset_v3  --arch cnn,rf,svm --dataset-real dataset_real_v3_out_of_grid  --activities FALL,IDLE,RUN,STAND,WALK,JUMP

# ── Confronto modalità di sottrazione (3 classi, rf+cnn) ────────────────────
#python csi_har_benchmark_pt.py --features doppler --arch cnn,rf --activities FALL,IDLE,WALK --subtract-mode none
#python csi_har_benchmark_pt.py --features doppler --arch cnn,rf --activities FALL,IDLE,WALK --subtract-mode dt
#python csi_har_benchmark_pt.py --features doppler --arch cnn,rf --activities FALL,IDLE,WALK --subtract-mode dt_ema --ema-alpha 0.80

# ── 5 classi, tutte le architetture ─────────────────────────────────────────
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py --features doppler --arch all --dataset-sim dataset_v2 --dataset-real dataset_real_v2 --activities FALL,IDLE,RUN,STAND,WALK

# ── EMA alpha sweep ──────────────────────────────────────────────────────────
#python csi_har_benchmark_pt.py --features doppler --arch rf --activities FALL,IDLE,WALK --subtract-mode dt_ema --ema-alpha 0.90
#python csi_har_benchmark_pt.py --features doppler --arch rf --activities FALL,IDLE,WALK --subtract-mode dt_ema --ema-alpha 0.95
#python csi_har_benchmark_pt.py --features doppler --arch rf --activities FALL,IDLE,WALK --subtract-mode dt_ema --ema-alpha 0.99

# ── [2026-07-02] Training su H_combined grezzo (H_s+H_d), nessuna sottrazione ──
# Cosa fa: usa --subtract-mode combined, che carica H_combined_v*.csv (canale
# grezzo, static+dynamic) invece di stimare H_s via DT/EMA e sottrarlo.
# Perché: le feature Doppler sono differenze temporali di H(t) e cancellano
# H_s per costruzione (dH_s/dt = 0), quindi il classificatore non vede mai
# H_s indipendentemente da come/se viene stimato. Questo esperimento verifica
# che addestrare direttamente sul canale grezzo dia risultati equivalenti a
# dt/dt_ema, senza il costo di stimare H_s tramite il Digital Twin.
#python csi_har_benchmark_pt.py --features doppler --arch all --activities FALL,IDLE,WALK --subtract-mode combined

# ── [2026-07-02] Dataset griglia v3, 6 classi, tutti i modelli ──────────────
# --arch all = CNN, LSTM, Transformer (deep learning) + RF, SVM (classici)
# — vedi _ALL_ARCHS in csi_har_benchmark_pt.py.

#ESPERIMENTO NLOS dataset_real_v3
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py --features doppler  --dataset-sim dataset_v3  --arch all --dataset-real dataset_real_v3  --activities FALL,IDLE,RUN,STAND,WALK,JUMP

#ESPERIMENTO LOS dataset_real_v4
#torchrun --nproc_per_node=4 csi_har_benchmark_pt.py --features doppler  --dataset-sim dataset_v3  --arch all --dataset-real dataset_real_v4  --activities FALL,IDLE,RUN,STAND,WALK,JUMP

# ── [2026-07-07] Generalizzazione a una stanza mai vista (unknown_room) ─────
# Cosa fa: allena su dataset_v3 (TTI-Lab, come sempre) e valuta zero-shot su
# dataset_unknown_room_target, una stanza con geometria completamente diversa
# (15m x 3m x 4m, tramezzo in posizione diversa, gNB/RIS/UE riposizionati;
# vedi run_acquisition_parallel_v2.py --scene-env unknown_real).
# Perché: gli altri esperimenti (dataset_real_v3/v4) cambiano solo l'arredo
# della STESSA stanza TTI-Lab; qui invece la stanza stessa è diversa, quindi
# la confusion matrix confrontata con quelle sopra isola quanto della
# generalizzazione già osservata regge anche su una geometria mai vista.
#ESPERIMENTO ROOM SCONOSCIUTA dataset_unknown_room_target
torchrun --nproc_per_node=4 csi_har_benchmark_pt.py --features doppler  --dataset-sim dataset_v3  --arch cnn --dataset-real dataset_unknown_room_target  --activities FALL,IDLE,RUN,STAND,WALK,JUMP --eval-only

