Per utilizzare lo script matlab, basta cambiare le seguenti variabili:
- filename: il nome del file .sc16 (COPIARE TUTTI I FILE (che sono dentro sniffer@10.8.9.16:/home/sniffer/spectrum_plotter/ NELLA CARTELLA DOVE GIRA LO SCRIPT)
- csi_symb_matlab: se stiamo parlando del csi in posizione 5 (33) o in posizione 9 (37)
- csi_spacing;k0_csi: se vogliamo vedere il csi a densità 3 (4;0) o se vogliamo vedere il csi a densità 1 (12;11)

Per ogni frame processato, lo script si fermerà a un keyboard (breakpoint) e verranno prodotti i 4 plot (costellazione pre e post equalizzazione e fasi pre e post equalizzazione). Se la costellazione equalizzata fa schifo, allora stiamo guardando il csi con la densità sbagliata. Per passare al frame successivo, basta continuare dopo il keyboard.

NB: il csi a densità 1 si trova un frame sì e un frame no. Stessa cosa il csi a densità 3. In particolare, se in quel frame si trova il csi a densità 1 non si trova quello a densità 3 e viceversa.

Per poter lavorare con i campioni equalizzati del csi, bisogna lavorare con la variabile Yeq, mentre il canale è H_est_csi1 subito sopra il keyboard.