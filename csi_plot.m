function csi_plot
	tic
	dir='./';
	filename='posizione1_20260330_142248.sc16';
	csi_symb_matlab = 33; % posizione 5
	% csi_symb_matlab = 37; %posizione 9
	% csi_spacing = 4;k0_csi=0; % density 3
	csi_spacing = 12;k0_csi=11; % density 1
	fs = 23.04e6;f0_MHz=1970;mu=0;L_max=8;nskip=0;BW_MHz=20;f0_corr=0;csi_known=false;
	if exist('csi_type','var')
		switch csi_type
			case 'srsRAN_4G'
				csi_known=true;
				csi1.sc_spacing=4;csi1.SFN=1;csi1.SFN_period=4;csi1.k0=0;csi1.bitmap=zeros(1,140);csi1.bitmap(1+[18 22 32 36])=1;csi1.nID=2;
				csi2.sc_spacing=12;csi2.SFN=0;csi2.SFN_period=8;csi2.k0=11;csi2.bitmap=zeros(1,140);csi2.bitmap(1+[18])=1;csi2.nID=2;
				% csi0.sc_spacing=12;csi0.SFN=-1;csi0.SFN_period=-1;csi0.k0=-1;csi0.bitmap=zeros(1,140);
			case 'srsRAN_project'
				csi_known=true;
				csi1.sc_spacing=4;csi1.SFN=1;csi1.SFN_period=2;csi1.k0=0;csi1.bitmap=zeros(1,140);csi1.bitmap(1+[32 36 46 50])=1;csi1.nID=1;
				csi2.sc_spacing=12;csi2.SFN=0;csi2.SFN_period=2;csi2.k0=11;csi2.bitmap=zeros(1,140);csi2.bitmap(1+[32])=1;csi2.nID=1;
				% csi0.sc_spacing=12;csi0.SFN=0;csi0.SFN_period=2;csi0.k0=8;csi0.bitmap=zeros(1,140);csi0.bitmap(1+[2*14+8])=1;
		end
	end
	% SIB1_known = false;
	SIB1_known = true;

	if strfind(filename,'sc16')
		sample_type='int16';sample_size=4;
	else
		sample_type='float32';sample_size=8;
	end

	cp_corr_thres = 0.7;
	PSS_corr_thres = 0.5; % 0.7?


	FIGURES = 0x0001; % synchronized frame & residual & eqsym

	fp = fopen([dir,filename],'r');
	if fp == -1
		error('Errore: file non aperto. Controlla il percorso e il nome del file.');
	end
	Ts = 1/fs;

	% parametri del frame
	Tf = 10e-3;
	Nsymb_slot = 14; % can be 12 for mu=2, with an extended N_CP
	Nsymb_subframe = Nsymb_slot*2^mu;
	Nsubframe_frame = 10;
	SCS = 15e3*2^mu;
	Nslot_frame = Nsubframe_frame*2^mu;
	Nsymb_frame = Nsymb_slot*Nslot_frame;

	% Ngrid_start,Ngrid_size: higher layer signaling
	if SIB1_known
		% 38.104 Table 5.3.2-1
		BW_MHz_tab = [...
			5  10  15  20  25  30  40  50  60  80  90 100];
		N_RB_tab = [...
			25  52  79 106 133 160 216 270  -1  -1  -1  -1; ...
			11  24  38  51  65  78 106 133 162 217 245 273; ...
			-1  11  18  24  31  38  51  65  79 107 121 135];
		N_RB = N_RB_tab(1+mu,BW_MHz==BW_MHz_tab);
		[FDA_BWP_int,nFDA_max] = FDA_BWP_int_calc(N_RB);
	else
		N_RB = -1; % will be discovered via the SIB1
	end

	ns = Tf*fs;
	ts = (0:ns-1)*Ts;

	% nello stato ACQUIRE cerchiamo il PBCH e, quando lo troviamo,
	% passiamo allo stato TRACK nel quale siamo allineati al frame e
	% abbiamo una idea abbastanza accurata del Cfo
	state = 'ACQUIRE';
	GSCN_MHz = GSCN_freqs_MHz(f0_MHz,fs/1e6,BW_MHz,SCS/1e6);
	N_GSCN_MHz = numel(GSCN_MHz);
	[hSSB,NhSSB,hSSB_decim] = SSB_filter(fs,SCS,ns);
	hSSB_state = zeros((NhSSB/hSSB_decim-1)*hSSB_decim,N_GSCN_MHz);
	phi_SSB = zeros(1,N_GSCN_MHz);
	if bitand(FIGURES,0x1000)
		GSCN_corr_log = zeros(3*N_GSCN_MHz,ns/hSSB_decim);
	end

	[hChannel,NhChannel] = Channel_filter(fs,BW_MHz);
	hChannel_state = zeros(NhChannel-1,1);

	% la frequenza di campionamento deve essere multipla di 128*2^mu*15kHz
	N_FFT = fs/SCS;
	if N_FFT ~= round(N_FFT)
		fprintf('The sampling frequency (fs) must be an integer multiple of %dMHz\n',128*2^mu*0.015);
		return
	end
	N_CP = 9/128*N_FFT;
	N_CP_07 = (9+2^mu)/128*N_FFT; % simboli 0 o 7*2^mu
	CP_slack = 4*N_FFT/128; % margine dal punto di inizio stimato del CP
	cp_len = N_CP*ones(1,Nsymb_subframe);
	cp_len(1+[0,7*2^mu]) = N_CP_07;
	sym_len=N_FFT+cp_len;
	% questo è necessario perché il segnale trasmesso non è semplicemente
	% Re{s(t)exp(2i*pi*f0_t)} ma ciascun simbolo viene sfasato dell'opposto
	% di questa quantità
	phase_comp = 2*pi*f0_MHz*1e6*(cp_len+[0,cumsum(sym_len(1:end-1))])*Ts;

	% -120 perché rispetto alla frequenza del GSCN, che corrisponde alla
	% portante 120 dell'SSB
	PSS_ind = (56:182) -120;
	SSB_ind = (0:239)  -120;
	PSS_freq = zeros(N_FFT,3);
	PSS_freq(1+N_FFT/2+PSS_ind,:) = PSS_func;
	PSS_time = ifft(ifftshift(PSS_freq,1));
	% PSS_time_CP = [PSS_time(end-N_CP+1:end,:);PSS_time];
	hPSS = conj(flipud(PSS_time(1:hSSB_decim:end,:)));
	hPSS_state = zeros(N_GSCN_MHz*3,size(hPSS,1)-1);
	hPSS_norm = ones(size(hPSS,1),1)*norm(hPSS(:,1))^2;
	hPSS_norm_state = zeros(N_GSCN_MHz,size(hPSS,1)-1);

	if ~csi_known
		% ricerca blind dei CSI#1 (density 3/12) e #2 (density 1/12)
		csi1.sc_spacing = 4;
		csi1.k0 = -1;
		csi1.bitmap = zeros(1,Nsymb_frame);
		csi1.SFN = -1;
		csi1.SFN_period = -1;
		csi1.nID = -1;
		csi2.sc_spacing = 12;
		csi2.k0 = -1;
		csi2.bitmap = zeros(1,Nsymb_frame);
		csi2.SFN = -1;
		csi2.SFN_period = -1;
		csi2.nID = -1;
	end

	fread(fp,[2 nskip],sample_type);

	% formato del CORESET#0, condiviso dai messaggi SIB e RandomAccess

	% main loop
	SSB_offset=CP_slack+1;k120=121;
	skip_for_sync = 0; % per la sincronizzazione di timing
	Cfo_eF = 0; % sincronizzazione di frequenza
	phi_s = 0;
	SFN = -1; % numero del frame, 10 bit
	initialSystemInfo.NFrame = SFN;
	alpha = 1; % contatore per la stima PID di Cfo
	fprintf('[ACQUIRE]\n');
	max_val = 0;
	try_again = 0;
	while true
		if ~try_again
			fseek(fp,skip_for_sync*sample_size,'cof'); % *8: float32
			phi_s = phi_s + 2*pi*(f0_corr-Cfo_eF)*skip_for_sync*Ts;
			[s_raw,ns_read] = fread(fp,[2 ns],sample_type);
			if ns_read < 2*ns
				fclose(fp);
				if numel(strfind(filename,'_seq'))==0
					break
				end
				% apri il prossimo file
				i=numel(filename);
				while filename(i)>='0' && filename(i)<='9'
					i=i-1;
				end
				filename(i+1:end)=num2str(mod(str2double(filename(i+1:end))+1,10^(numel(filename)-i)),'%04d');
				if ~exist([dir,filename],'file')
					break
				end
				fp=fopen([dir,filename],'r');
				tmp = fread(fp,[2 ns-ns_read/2],sample_type);
				s_raw = [s_raw,tmp];
			end
			[s,hChannel_state] = filter(hChannel,1,[1 1i]*s_raw,hChannel_state);
			% f0_corr_SSB è usata per correggere la traccia nel caso che
			% non sia stata acquisita centrata intorno alla portante
			s = s .* exp(2i*pi*(f0_corr-Cfo_eF)*ts + 1i*phi_s);
			phi_s = phi_s + 2*pi*(f0_corr-Cfo_eF)*ns*Ts;
			clear s_raw
		end

		% UL
		%imagesc(abs(whole_frame_fft(circshift(s,[0 -917]),2048,140,0,zeros(1,140),0x0000)))
		% if strcmp(state,'TRACK')
		% 	% 7c
		% 	tmp = whole_frame_fft(circshift(s,[0 468]),1024,280,0,zeros(1,280),0x0000);
		% 	tmp(513,:) = 0;
		% 	cf(13)
		% 	subplot(2,1,1)
		% 	imagesc(abs(tmp))
		% 	subplot(2,1,2)
		% 	imagesc(angle(tmp.^4))
		% 	display(i)
		% 	drawnow
		% 	keyboard
		% end

		if SFN >= 0
			fprintf('SFN:%d\n',SFN);
		else
			fprintf('SFN:unk\n');
		end

		switch state
			case 'ACQUIRE'
				
				% max_val = 0;
				for i = 1:N_GSCN_MHz
					f_SSB = (GSCN_MHz(i)-f0_MHz)*1e6;
					s_tmp=0;
					for j = 1:hSSB_decim
						[s_tmp_decim,hSSB_state((j-1)*(NhSSB/hSSB_decim-1)+(1:NhSSB/hSSB_decim-1),i)] = ...
							filter(hSSB(hSSB_decim+1-j:hSSB_decim:end),1,...
							s(j:hSSB_decim:end).*exp(-2i*pi*f_SSB*ts(j:hSSB_decim:end)+1i*phi_SSB(i)),...
							hSSB_state((j-1)*(NhSSB/hSSB_decim-1)+(1:NhSSB/hSSB_decim-1),i));
						s_tmp=s_tmp+s_tmp_decim;
					end
					% s_tmp_ref=filter(hSSB,1,s.*exp(-2i*pi*5e6*ts+1i*phi_SSB(i)));
					phi_SSB(i) = phi_SSB(i) - 2*pi*f_SSB*ns*Ts;
					% verifica nello stato zero
					% norm(s_tmp-s_tmp_ref(hSSB_decim:hSSB_decim:end))
					[tmp_den,hPSS_norm_state(i,:)] = filter(hPSS_norm,1,...
						abs(s_tmp).^2,hPSS_norm_state(i,:));
					for j = 1:3
						[tmp,hPSS_state((i-1)*3+j,:)] = filter(hPSS(:,j),1,...
							s_tmp,hPSS_state((i-1)*3+j,:));
						tmp = tmp./sqrt(tmp_den); % normalizzata
						if bitand(FIGURES,0x1000)
							GSCN_corr_log(3*(i-1)+j,:) = abs(tmp);
						end
						% FIXME? (normalizzare la correlazione)
						[val,pos]=max(abs(tmp));
						% cf(2);plot(abs(tmp))
						% keyboard
						if val > max_val
							max_val = val;
							max_pos = pos;
							GSCN_freq = GSCN_MHz(i);
							% la frequenza GSCN è della portante 120
							k120 = round((GSCN_freq-f0_MHz)*1e6/SCS);
							NID2 = j-1;
							SSB_offset = 1+(max_pos)*hSSB_decim-N_FFT-N_CP-(NhSSB/hSSB_decim-1)/2*hSSB_decim;
						end
					end
				end

				Y_PBCH = zeros(240,4);
				cp_corr = zeros(1,4);
				for i = 0:3
					% ultimo parametro: cfo_comp per la correzione di Cfo simbolo per simbolo
					[tmp,tmp_corr] = ofdm_fft(s,SSB_offset+i*(N_FFT+N_CP),CP_slack,N_CP,N_FFT,0); % 1?
					Y_PBCH(:,1+i) = tmp(1+N_FFT/2+k120+SSB_ind)*exp(1i*phase_comp(2+i));
					cp_corr(1+i) = tmp_corr;
				end
				if min(abs(cp_corr)) > cp_corr_thres
					% questa stima di canale verrà usata per la stima dell'errore di timing
					H_est_PSS = Y_PBCH(1+120+PSS_ind,1+0)...
						.*PSS_freq(1+N_FFT/2+PSS_ind,1+NID2);
					SSS_freq = SSS_func(NID2);
					[PSS_SSS_corr,SSS_ind]=max(abs((Y_PBCH(1+120+PSS_ind,1+2))'*(SSS_freq.*H_est_PSS)));
					if PSS_SSS_corr/(norm(Y_PBCH(1+120+PSS_ind,1+2))*norm(SSS_freq(:,SSS_ind).*H_est_PSS)) < PSS_corr_thres
						continue
					end
					fprintf('  [PSS/SSS] GSCN: %gMHz ',GSCN_freq);

					NID1 = SSS_ind-1;
					H_est_SSS = Y_PBCH(1+120+PSS_ind,1+2)...
						.*SSS_freq(:,1+NID1);
					% angle(sum(H_est_SSS.*conj(H_est_PSS)))/2==2*pi*delta_f*(N_FFT+N_CP)*Ts
					Cfo_est_PBCH=angle(sum(H_est_SSS.*conj(H_est_PSS)))/(4*pi*(N_FFT+N_CP)*Ts);

					ncellid = 3*NID1+NID2;
					Cfo_est_CP = mean(angle(cp_corr))/(2*pi*N_FFT*Ts);
					Kpf=0.1;Kif=0.2;
					%Kpf=0.1;Kif=0.05;
					Cfo_eD = 0;
					% come stima iniziale preferiamo la stima basata sul CP (maggiore dinamica)
					Cfo_eF2 = 0*Cfo_est_PBCH+1*Cfo_est_CP;
					Cfo_eF = Kpf*Cfo_eD+Cfo_eF2;
					fprintf('ncellid: %d Cfo: %.1fkHz(CP) %.1fkHz(PSS->SSS)',...
						ncellid,Cfo_est_CP*1e-3,Cfo_est_PBCH*1e-3);

					% stima iniziale della varianza di rumore nei resource
					% element inutilizzati dal PBCH
					noise_est = mean(abs(Y_PBCH(1+[0:49,189:239],1+0)).^2);
					noise_est_dB = 10*log10(noise_est);

					Ycorr = unwrap_QPSK(Y_PBCH(:,1+1)); % il DMRS è nei simboli 1, 2 e 3
					dmrs_ind = mod(ncellid,4):4:239;
					C_est = Y_PBCH(1+dmrs_ind,1+1)./Ycorr(1+dmrs_ind);
					%C_est = DMRS_heuristic(Y_PBCH(:,1+1),dmrs_ind,C_est);
					[is_ok,c_init_DMRS] = DMRS_process(C_est,0,4);
					if is_ok
						% c_init_DMRS ==  2^11*(iSSB+1)*(floor(ncellid/4)+1) ...
						%               + 2^6*(iSSB+1)
						%               + mod(ncellid,4)
						iSSB1 = floor(c_init_DMRS/2^11)/(1+floor(ncellid/4))-1;
						iSSB2 = floor(mod(c_init_DMRS,2^11)/2^6)-1;
						%c = scrambling(288,c_init_DMRS,1);
						%C = (1-2*c(1:2:end))+1i*(1-2*c(2:2:end));
						%Yeq = PBCH_equalize(Y,dmrs_ind,C,noise_est)
						if iSSB1 == iSSB2
							iSSB = iSSB1;
							fprintf(' iSSB: %d\n',iSSB);
						else
							keyboard
						end
					else
						dmrsIndices = nrPBCHDMRSIndices(ncellid);
						dmrsEst = zeros(1,8);
						for ibar_SSB = 0:7
							refGrid = zeros([240 4]);
							refGrid(dmrsIndices) = nrPBCHDMRS(ncellid,ibar_SSB);
							[hest,nest] = nrChannelEstimate(Y_PBCH,refGrid,'AveragingWindow',[0 1]);
							dmrsEst(ibar_SSB+1) = 10*log10(mean(abs(hest(:).^2)) / nest);
						end
						iSSB = find(dmrsEst==max(dmrsEst)) - 1;
						fprintf(' iSSB(*): %d\n',iSSB);
					end

					% "riavvolgi" al primo file dopo l'acquisizione dell'SSB
					if numel(strfind(filename,'_seq'))
						fclose(fp);
						i=numel(filename);
						while filename(i)>='0' && filename(i)<='9'
							i=i-1;
						end
						filename(i+1:end)='0';
						fp=fopen([dir,filename],'r');
					else
						frewind(fp);
					end
					fread(fp,[2 nskip],sample_type);
					switch mu
						case 0
							% 38.211
							symSSB_tab = [ 2  8 16 22]; % Case A
						case 1
							symSSB_tab = [ 2  8 16 22 30 36 44 50]; % Case C
						otherwise
							% FIXME: other SSB Cases
							keyboard
					end
					symSSB_ofs = symSSB_tab(1+iSSB);
					fread(fp,[2 mod(SSB_offset - symSSB_ofs*(N_FFT+N_CP) ...
						- ceil(symSSB_ofs/(7*2^mu))*(N_CP_07-N_CP),fs*10e-3)],sample_type);
					%keyboard
					%tmp = fread(fp,[2 NhChannel-1],sample_type);
					%[~,hChannel_state] = filter(hChannel,1,[1 1i]*tmp);
					state = 'TRACK';
					fprintf('[TRACK]\n');
					SSB_offset = 0;
				else
					Cfo_eD = 0;
				end

			case 'TRACK'

				Y = whole_frame_fft(s,N_FFT,Nsymb_frame,mu,phase_comp,0x0040);
				Yr = abs(Y);
				if bitand(FIGURES,0x0001)
					currentfigure(1)
					[H_est_csi1,sc] = csi_H_est(Y,csi_symb_matlab,k0_csi,csi_spacing,N_RB,1);
					
					pilot_idx = 1 + N_FFT/2 + sc;
					H_pilot = H_est_csi1(pilot_idx);

					Yeq = Y(pilot_idx,csi_symb_matlab) ./ H_pilot;
					% Yeq_corr=Yeq;
					subplot(2,2,1)
					plot(real(Y(pilot_idx,csi_symb_matlab)), imag(Y(pilot_idx,csi_symb_matlab)), 'r*');
					title('Received Signal Constellation');
					subplot(2,2,2)
					plot(real(Yeq), imag(Yeq), '*')
					title('Equalized Signal Constellation');
					subplot(2,2,3)
					plot(angle(Y(pilot_idx,csi_symb_matlab)), 'ro')
					title('Received Signal Phase');
					subplot(2,2,4)
					plot(angle(Y(pilot_idx,csi_symb_matlab) ./ H_pilot), 'o')
					title('Equalized Signal Phase');
					% axis equal; grid on
					% xlim([-3 3]); ylim([-3 3])
					keyboard
				end
				
				[PBCH_in_frame,skip_for_sync,Cfo_eD] = PSS_timing_offset_est(Y,k120,...
					PSS_ind,symSSB_ofs,H_est_PSS,PSS_freq,SSS_freq,ncellid,PSS_corr_thres,fs,FIGURES);
				if PBCH_in_frame
					noise_est_new = mean(abs(Y(1+N_FFT/2+k120-120+([0:49,189:239]),1+symSSB_ofs)).^2);
					% stima AR
					if (noise_est_new/noise_est+noise_est/noise_est_new)>3
						noise_est = noise_est_new;
					else
						noise_est = 0.9*noise_est + 0.1*noise_est_new;
					end
					noise_est_dB = 10*log10(noise_est);

					% zero the PBCH in Yr
					Yr(1+N_FFT/2+k120-120+(56:182),symSSB_ofs+1)=0;
					Yr(1+N_FFT/2+k120-120+(0:239),symSSB_ofs+2)=0;
					Yr(1+N_FFT/2+k120-120+([0:47,56:182,192:239]),symSSB_ofs+3)=0;
					Yr(1+N_FFT/2+k120-120+(0:239),symSSB_ofs+4)=0;
				end
				if ~PBCH_in_frame
					if initialSystemInfo.NFrame >= 0
						initialSystemInfo.NFrame = initialSystemInfo.NFrame + 1;
					end
					alpha = alpha + 1; % tempo dall'ultima misura
				else
					Cfo_eF2 = Cfo_eF2 + Kif*alpha*Cfo_eD;
					Cfo_eF = Kpf*alpha*Cfo_eD+Cfo_eF2;
					alpha = 1; % tempo dall'ultima misura
					%fprintf('Cfo_eD: %g Cfo_eF:%g [Hz]\n',Cfo_eD,Cfo_eF);

					% demodula il PBCH
					dmrsIndices = nrPBCHDMRSIndices(ncellid);
					pssIndices = nrPSSIndices;
					sssIndices = nrSSSIndices;
					refGrid = zeros(240,4);
					refGrid(pssIndices) = nrPSS(ncellid);
					refGrid(dmrsIndices) = nrPBCHDMRS(ncellid,iSSB);
					refGrid(sssIndices) = nrSSS(ncellid);
					Y_PBCH = Y(1+N_FFT/2+k120+SSB_ind,symSSB_ofs+(1:4));
					[hest,nest,hestInfo] = nrChannelEstimate(Y_PBCH,refGrid,'AveragingWindow',[0 1]);
					[pbchIndices,pbchIndicesInfo] = nrPBCHIndices(ncellid);
					pbchRx = nrExtractResources(pbchIndices,Y_PBCH);
					pbchHest = nrExtractResources(pbchIndices,hest);
					[pbchEq,csi] = nrEqualizeMMSE(pbchRx,pbchHest,nest);
					Qm = pbchIndicesInfo.G / pbchIndicesInfo.Gd; % 2:QPSK per il PBCH
					csi = repmat(csi.',Qm,1);csi = reshape(csi,[],1);
					if L_max == 4,ssbIndex = mod(iSSB,4);else ssbIndex = iSSB;end;
					pbchLLR = nrPBCHDecode(pbchEq,ncellid,ssbIndex,nest) .* csi;
					polarListLength = 8;
					[~,crcBCH,trblk,sfn4lsb,nHalfFrame,msbidxoffset] = ...
						nrBCHDecode(pbchLLR,polarListLength,L_max,ncellid);
					%fprintf('crc:%d BCCH-BCH-Message:%s SFN%%16:%d hf:%d msbidxoffset:%d\n',crcBCH,...
					%	char(trblk+'0'),2.^(3:-1:0)*sfn4lsb,nHalfFrame,msbidxoffset);
					if crcBCH == 0
						if L_max==64 % FR2?
							ssbIndex = ssbIndex + (bit2int(msbidxoffset,3) * 8);
							k_SSB = 0;
						else
							k_SSB = msbidxoffset * 16;
						end
						mib = fromBits(MIB,trblk(2:end));
						initialSystemInfo = initSystemInfo(mib,sfn4lsb,k_SSB,L_max);
						%disp('BCH/MIB Content:')
						%disp(initialSystemInfo);
						fprintf('  [PBCH] SCS:%d kSSB:%d\n',...
							initialSystemInfo.SubcarrierSpacingCommon,...
							initialSystemInfo.k_SSB);
						% if ~isCORESET0Present(BlockPattern,initialSystemInfo.k_SSB)
						% 	fprintf('CORESET 0 is not present (k_SSB > k_SSB_max).\n');
						% 	keyboard
						% end
						SFN = initialSystemInfo.NFrame;
					else
						%fprintf('BCH CRC fail.\n')
						keyboard
					end
				end
		end
		if ~try_again
			if SFN >= 0
				SFN = mod(SFN+1,1024);
			end
		end
	end
	toc
end


% calcola le BWPsize possibili per un dato N_RB
function [FDA_BWP_int,nFDA_max] = FDA_BWP_int_calc(N_RB)
	% una BWP deve potere contenere il DCI e la banda minima di un DCI si
	% ottiene con AL 1 e duration 3, 2 RB
	N_RB_min = 1;
	nFDA_max = ceil(log2(N_RB*(N_RB+1)/2));
	FDA_BWP_int = zeros(1+nFDA_max,2);
	for i = N_RB_min:N_RB
		n = ceil(log2(i*(i+1)/2));
		if FDA_BWP_int(1+n,1)==0
			FDA_BWP_int(1+n,:) = i;
		else
			FDA_BWP_int(1+n,2) = i;
		end
	end
end

function [H_est, sc] = csi_H_est(Y, l_matlab, k0, csi_sc_spacing, N_RB, nID)
    Nsc_RB = 12;
    N_FFT = size(Y,1);
    % keyboard
    % l_matlab è 1-based (es. 33), convertiamo in 0-based
    l = l_matlab - 1;
    n_s_f      = floor(l / 14);   % slot nel frame
    l_in_slot  = mod(l, 14);      % simbolo nello slot
    
    % FIX 1: indici subportanti corretti con k0
    sc = k0 + (0:csi_sc_spacing:N_RB*Nsc_RB-1) - N_RB*Nsc_RB/2;
    
    Ycsi = Y(1 + N_FFT/2 + sc, l_matlab);
    
    % FIX 2: c_init corretto secondo 38.211
    c_init_CSI = mod(2^10 * (14*n_s_f + l_in_slot + 1) * (2*nID + 1) + nID, 2^31);
    
    M_PN = 2*numel(sc);
    c = scrambling(M_PN, c_init_CSI).';
    C_csi = ((1 - 2*c(1:2:end)) + 1i*(1 - 2*c(2:2:end))); %/sqrt(2);
    % Stima grezza sui pilot
    H_est = zeros(N_FFT, 1);
    H_est(1 + N_FFT/2 + sc) = Ycsi ./ C_csi;
    
    % Interpolazione via IFFT/finestra/FFT
    temp = ifftshift(ifft(fftshift(H_est)));
    len  = floor(N_FFT / (2*csi_sc_spacing));
    temp(N_FFT/2 + (-len:len)) = temp(N_FFT/2 + (-len:len)) .* gausswin(2*len+1, 1);
    temp([1:N_FFT/2-len-1, N_FFT/2+len+1:end]) = 0;
    H_est_interp = fftshift(fft(ifftshift(temp))) * csi_sc_spacing;
	% fit lineare: phase ≈ a*k + b
	
    H_est = H_est_interp;
end

function [PBCH_in_frame,skip_for_sync,Cfo_est] = PSS_timing_offset_est(Y,k120,PSS_ind,...
		symSSB_ofs,H_est_PSS0,PSS_freq,SSS_freq,ncellid,PSS_corr_thres,fs,FIGURES)
	N_FFT = size(Y,1);
	NID2 = mod(ncellid,3); 
	NID1 = (ncellid - NID2)/3;

	% stima di timing basata sul PSS equalizzato (SNR alto).
	% se il PSS non è presente, val è basso e skip_for_sync=0
	exp_iphi_epst = Y(1+N_FFT/2+k120+PSS_ind,1+symSSB_ofs) ...
		./(H_est_PSS0.*PSS_freq(1+N_FFT/2+PSS_ind,1+NID2)); % *conj?
	tmp = fftshift(ifft(exp_iphi_epst,512))*512/127;
	[val,pos]=max_parab(abs(tmp));pos=(pos-1-512/2)/(512/127);
	PBCH_in_frame = val > PSS_corr_thres;
	if val < PSS_corr_thres
		skip_for_sync = 0;
		Cfo_est = 0;
	else
		Ts_tmp = 1/(fs/(127*15e3));
		skip_for_sync = round(pos/Ts_tmp);
		H_est_PSS = Y(1+N_FFT/2+k120+PSS_ind,1+symSSB_ofs)./PSS_freq(1+N_FFT/2+PSS_ind,1+NID2);
		H_est_SSS = Y(1+N_FFT/2+k120+PSS_ind,3+symSSB_ofs)./SSS_freq(:,1+NID1);
		N_CP=9/128*N_FFT;
		Cfo_est = angle(sum(H_est_SSS([1:63,67:127]).*conj(H_est_PSS([1:63,67:127]))))*fs/(4*pi*(N_FFT+N_CP));
	end
end

function Y = whole_frame_fft(s,N_FFT,Nsymb_frame,mu,phase_comp,FIGURES)
	ns = numel(s);
	N_CP = 9/128*N_FFT;
	N_CP_07 = (9+2^mu)/128*N_FFT;
	CP_slack = N_FFT/128;
	Nsymb_subframe = Nsymb_frame/10;

	ofs = 0;
	l = 0;
	i = 0;
	Y = zeros(N_FFT,Nsymb_frame);
	Ycp_corr = zeros(1,Nsymb_frame);
	while i < Nsymb_frame

		if bitand(FIGURES,0x0004) && (ofs > 0 && ofs < ns-2*N_CP_07-N_FFT)
			cp_corr_log = zeros(4*CP_slack+1,2);
			for k = -2*CP_slack:2*CP_slack
				s1 = s(ofs+k      +(CP_slack+1:N_CP-CP_slack));
				s2 = s(ofs+k+N_FFT+(CP_slack+1:N_CP-CP_slack));
				cp_corr_log(1+2*CP_slack+k,:)=[k,s2*s1'/(norm(s1)*norm(s2))];
			end
			currentfigure(3);
			hold off
			plot(cp_corr_log(:,1),abs(cp_corr_log(:,2)))
			hold on
			plot(CP_slack*[1 1],[0 1.05],'k:')
			plot(-CP_slack*[1 1],[0 1.05],'k:')
			plot(2*CP_slack*[-1 1],[1 1],'k--')
			ylabel('|normalized CP correlation|')
			xlabel('delay')
			ylim([0 1.05])
			xlim(2*CP_slack*[-1 1])
			drawnow
		end

		if mod(l,7*2^mu) == 0
			[Y(:,1+i),Ycp_corr(1+i)] = ofdm_fft(s,ofs+N_CP_07-N_CP,CP_slack,N_CP,N_FFT,0);
			Y(:,1+i) = Y(:,1+i)*exp(1i*phase_comp(1+l));
			ofs = ofs + N_FFT + N_CP_07;
		else
			[Y(:,1+i),Ycp_corr(1+i)] = ofdm_fft(s,ofs,CP_slack,N_CP,N_FFT,0);
			Y(:,1+i) = Y(:,1+i)*exp(1i*phase_comp(1+l));
			ofs = ofs + N_FFT + N_CP;
		end
		i = i+1;
		l = mod(l+1,Nsymb_subframe);
	end

end

% cfo_comp: per-symbol Cfo compensation based on the phase of the cyclic
%           prefix correlation
function [Y,cp_corr]=ofdm_fft(s,ofs,CP_slack,N_CP,N_FFT,cfo_comp)
	s1 = s(ofs+      (CP_slack+1:N_CP-CP_slack));
	s2 = s(ofs+N_FFT+(CP_slack+1:N_CP-CP_slack));
	cp_corr = s2*s1'/(norm(s1)*norm(s2));
	if cfo_comp
		dphi_est = angle(cp_corr)/N_FFT;
		Y = fftshift(fft(s(ofs+N_CP-CP_slack+(1:N_FFT)) ...
			.*exp(-1i*dphi_est*(-CP_slack:N_FFT-CP_slack-1))))...
			.*exp(2i*pi*(-N_FFT/2:N_FFT/2-1)*CP_slack/N_FFT);
	else % cfo_comp==0
		Y = fftshift(fft(s(ofs+N_CP-CP_slack+(1:N_FFT))))...
			.*exp(2i*pi*(-N_FFT/2:N_FFT/2-1)*CP_slack/N_FFT);
	end
end

function GSCN = GSCN_freqs_MHz(f0_MHz,fs_MHz,BW_MHz,SCS_MHz)
	% 38.104 5.4.3
	if BW_MHz == 3
		keyboard
		%BW=3MHz
		% #GSCN = 26638+3N+(M-3)/2
		% 0-1GHz: N*0.6MHz+M*50kHz, N=1:1665, M={1,3,5}
		% Band n100
		% #GSCN=41637 920.73MHz
		% #GSCN=41638 921.45MHz
	else % BW>3MHz
		if f0_MHz < 3000
			% #GSCN = 3*N+(M-3)/2
			% 0-3GHz: N*1.2MHz + M*0.05MHz, N=1:2499 M={1,3,5}

			% N_min*1.2+0.05 - 120*SCS_MHz >= f0_MHz-BW_MHz/2
			N_min = ceil((f0_MHz-BW_MHz/2 + 120*SCS_MHz - 0.05)/1.2);
			% N_max*1.2+0.25 + 119*SCS_MHz <= f0_MHz+BW_MHz/2
			N_max = floor((f0_MHz+BW_MHz/2 - 119*SCS_MHz - 0.25)/1.2);
			for M = [1 3 5]
				GSCN = (N_min:N_max)*1.2 + M*0.05;
				if abs(mod((GSCN(1)-f0_MHz)/SCS_MHz+0.5,1)-0.5)<0.05
					break
				end
			end
			% k_GSCN_vec = round((GSCN-f0_MHz)/SCS_MHz);
		elseif f0_MHz<24250
			% #GSCN = 7499+N
			% 3-24.25GHz: 3GHz + N*1.44MHz, N=0:14756

			% 3000+N_min*1.44 - 120*SCS_MHz >= f0_MHz-BW_MHz/2
			N_min = ceil((f0_MHz-BW_MHz/2 - 3000 + 120*SCS_MHz)/1.44);
			% 3000+N_max*1.44 + 119*SCS_MHz <= f0_MHz+BW_MHz/2
			N_max = floor((f0_MHz+BW_MHz/2 - 3000 - 119*SCS_MHz)/1.44);
			% if abs(f0_MHz - 3760) < 0.72
			% 	GSCN = 3731.52;
			% elseif abs(f0_MHz - 3680) < 0.72
			% 	GSCN = 3649.44;
			% elseif abs(f0_MHz - 3630) < 0.72
			% 	GSCN = 3624.96;
			% elseif abs(f0_MHz - 3590) < 0.72
			% 	GSCN = 3570.24;
			% elseif abs(f0_MHz - 3900) < 0.72
			% 	GSCN = 3900;
			% else
			GSCN = 3000+(N_min:N_max)*1.44;
			% GSCN = GSCN(4:6);
			% end
		else
			keyboard
			% #GSCN = 22256+N
			% 24.25-100GHz: 24.25008GHz + N*17.28MHz, N=22256:26639
		end
	end
end

function d_PSS = PSS_func
	x = zeros(1,127);
	x(1+(0:6)) = [0 1 1 0 1 1 1];
	for i = 8:127
		x(i)=mod(x(i-7)+x(i-3),2);
	end
	d_PSS = zeros(127,3);
	for NID2 = 0:2
		d_PSS(:,1+NID2) = 1-2*x(1+mod((0:126)+43*NID2,127));
	end
end

function d_SSS = SSS_func(NID2)
	x0 = zeros(1,127);
	x0(1+(0:6)) = [1 0 0 0 0 0 0];
	x1 = zeros(1,127);
	x1(1+(0:6)) = [1 0 0 0 0 0 0];
	for i = 8:127
		x0(i)=mod(x0(i-3)+x0(i-7),2);
		x1(i)=mod(x1(i-6)+x1(i-7),2);
	end
	d_SSS = zeros(127,336);
	for NID1 = 0:335
		m0 = 15*floor(NID1/112)+5*NID2;
		m1 = mod(NID1,112);
		d_SSS(1+(0:126),1+NID1) = (1-2*x0(1+mod((0:126)+m0,127))).*(1-2*x1(1+mod((0:126)+m1,127)));
	end
end

function [hChannel,NhChannel] = Channel_filter(fs,BW_MHz)
	if 0
		hChannel = 1;
		NhChannel = 1;
	elseif 1 % FIRPM
		rp = 1;           % Passband ripple in dB
		rs = 50;          % Stopband ripple in dB
		BW = BW_MHz*1e6;
		%f = [BW/2 BW/2+450e3];
		f = [BW/2 BW/2+1e6];  % Cutoff frequencies
		a = [1 0];        % Desired amplitudes
		dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
		[n,fo,ao,w] = firpmord(f,a,dev,fs);
		hChannel = firpm(n,fo,ao,w);
	else
		a = 0.18;
		f = [(1-a/2)*BW_MHz/2 (1+a/2)*BW_MHz/2+450e3];
		a = (f(2)-f(1))/(f(2)+f(1));
		LChannel = 31;
		hChannel = raised_cosine(a,(-LChannel:LChannel)/fs,1/fs);
	end
	NhChannel = numel(hChannel);
end

function [hSSB,NhSSB,decim] = SSB_filter(fs,SCS,ns)
	decim = floor(fs/(240*SCS));
	while mod(ns,decim) ~= 0
		decim = decim - 1;
	end
	if 0 % FIRPM
		rp = 1;           % Passband ripple in dB
		rs = 20;          % Stopband ripple in dB
		f = [120*SCS 120*SCS+0.055*fs];
		a = [1 0];        % Desired amplitudes
		dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)];
		[n,fo,ao,w] = firpmord(f,a,dev,fs);
		hSSB = firpm(n,fo,ao,w);
		NhSSB = numel(hSSB);
	else % RCOS
		fs_decim = fs/decim;
		f = [120*SCS fs_decim/2];
		a = (f(2)-f(1))/(f(2)+f(1));
		while a < 0.16
			decim = decim - 1;
			while mod(ns,decim) ~= 0
				decim = decim - 1;
			end
			fs_decim = fs/decim;
			f = [120*SCS fs_decim/2];
			a = (f(2)-f(1))/(f(2)+f(1));
		end
		LSSB = 6;
		hSSB = raised_cosine(a,(-LSSB*decim:LSSB*decim+decim-1)/fs,decim/fs)/decim;
		% hold off
		% plot((0:4095)*fs/4096,20*log10(abs(fft(hSSB,4096))));
		% hold on
		% plot([0 1 1]*SCS*120,[0 0 -40]);
		% plot([fs_decim,fs_decim,fs]/2,[0 -40 -40])
		% keyboard
		NhSSB = (2*LSSB+1)*decim;
	end
end
function s=raised_cosine(a,t,T)
	t = t/T;
	s = (1-a)/2*sinc((1-a)*t) + ...
		(1+a)/2*sinc((1+a)*t) + ...
		a/2*sin(pi*t) .* (sinc(a*t-1/2)-sinc(a*t+1/2));
end

function currentfigure(nf)
	if ishghandle(nf)
		set(0,'CurrentFigure',nf)
	else
		figure(nf)
	end
	set(nf,'WindowStyle','docked')
end

function Y_corr = unwrap_QPSK(Y)
	N = numel(Y);
	Y_corr = zeros(size(Y)); %correggo la fase prendendo come riferimento il primo simbolo
	%affinché sia compresa tra pi/4 e -pi/4 (intorno a zero)
	Y_corr(1) = Y(1);
	for i = 2:N
		%differenza di fase tra il simbolo i-esimo reale e quello i-1 esimo corretto
		tmp = angle(Y(i)*conj(Y_corr(i-1)));
		if abs(tmp) > 3*pi/4
			Y_corr(i) = -Y(i); %aggiusto di 180°
		elseif tmp < -pi/4
			Y_corr(i) = 1i*Y(i); %aggiusto di +90°
		elseif tmp > pi/4
			Y_corr(i) = -1i*Y(i); %aggiusto di -90°
		else
			Y_corr(i) = Y(i);
		end
	end
end

function [is_ok,c_init_DMRS,C_est_DMRS] = DMRS_process(C_est,k0,sc_spacing)
	M_PN = numel(C_est)*2;
	c_est = zeros(1,M_PN);
	Nc = 1600;
	% i DMRS del PDCCH e PDSCH, e le sequenze CSI, sono generate con
	% l'indice k valutato a partire dall'inizio della BandwidthPart (non a
	% partire dall'inizio della parte utilizzata del simbolo), quindi
	% dobbiamo allineare la sequenza introducendo un ritardo di:
	% - floor(k0/Nsc_RB)*Nsc_RB (numero di resource block prima di quello
	%   dal quale abbiamo estratto i DMRS)
	% - /DMRSperRB (è il numero di portanti DMRS per RB)
	% - *2 perché è modulato in QPSK, 2 bit per simbolo
	Nsc_RB = 12;
	D = floor(k0/Nsc_RB)*Nsc_RB/sc_spacing*2;
	x2_est = zeros(1,Nc+D+M_PN);
	%ricostruzione stati iniziali
	for i = 1:4
		C_est_DMRS = C_est*exp(1i*(pi/4+(i-1)*pi/2));
		c_est(1+(0:2:M_PN-1)) = real(C_est_DMRS)<0;
		c_est(1+(1:2:M_PN-1)) = imag(C_est_DMRS)<0;
		x2_est(Nc+D+(1:M_PN)) = mod(c_est+scrambling_x1(M_PN,D),2); %xor con x1 presunta
		is_ok = true;
		%x2 soddisfa la ricorsione?
		for n = Nc+D+M_PN-31:-1:Nc+D+1
			if x2_est(n) ~= mod(x2_est(n+31)+x2_est(n+3)+x2_est(n+2)+x2_est(n+1),2)
				is_ok = false;
				break
			end
		end
		if is_ok
			break
		end
	end
	if M_PN > 31
		for n = Nc+D:-1:1
			x2_est(n) = mod(x2_est(n+31)+x2_est(n+3)+x2_est(n+2)+x2_est(n+1),2);
		end
	end
	c_init_DMRS = bi2de(x2_est(1+(0:30)));
	% if is_ok
	% 	% verifica
	% 	c = scrambling(M_PN+D,c_init_DMRS);
	% 	fprintf('k0:%d D:%d |c-c_est|:%d DMRSperRB:%d\n',...
	% 	k0,D,sum(abs(c(end-M_PN+1:end)-c_est)),DMRSperRB);
	% end
end

% 5.2.1 38.211
function c = scrambling(M_PN,c_init,x1_init)
	if nargin < 3
		x1_init = 1;
	end
	if nargin < 2
		% N_symb^slot = 14 tranne per mu=2
		% n_{s,f}^mu = numero slot nel frame (per dato mu) {0-9 per mu=0}
		% l = OFDM symbol number within the slot {0-13 per mu=0}
		% N_ID: pdcch-DMRS-ScramblingID
		% c_init = mod(2^17*(N_symb^slot*n_{s,f}^mu+l+1)*(2*N_ID+1)+2*N_ID,2^31);
		c_init = 1;
	end
	Nc = 1600;
	x1 = zeros(1,Nc+M_PN);
	x2 = zeros(1,Nc+M_PN);
	x1(1+ 0) = x1_init;
	x1(1+ (1:30)) = 0;
	x2(1+ (0:30)) = de2bi(c_init,31);
	for n = 1:M_PN+Nc-31
		x1(n+31) = mod(x1(n+3)+x1(n),2);
		x2(n+31) = mod(x2(n+3)+x2(n+2)+x2(n+1)+x2(n),2);
	end
	c = zeros(1,M_PN);
	for n = 1:M_PN
		c(n) = mod(x1(n+Nc)+x2(n+Nc),2);
	end
end

function x1_subseq = scrambling_x1(M_PN,D)
	if nargin < 2
		D = 0;
	end
	Nc = 1600+D;
	x1 = zeros(1,Nc+M_PN);
	x1(1+ 0) = 1;
	x1(1+ (1:30)) = 0;
	for n = 1:M_PN+Nc-31
		x1(n+31) = mod(x1(n+3)+x1(n),2);
	end
	x1_subseq = x1(Nc+(1:M_PN));
end

function [y0_est,x0_est] = max_parab(y,x)
	n = numel(y);
	[~,pos] = max(y);
	if nargin < 2
		x = 1:n;
		dx = 1;
	else
		dx = x(2)-x(1);
	end
	x0 = x(pos);
	y0 = y(pos);
	xm = x(1+mod(pos-2,n));
	ym = y(1+mod(pos-2,n));
	xp = x(1+mod(pos,n));
	yp = y(1+mod(pos,n));
	corr_x = (yp-ym)/2/(2*y0-ym-yp);
	if isnan(corr_x)
		x0_est = x0;
		y0_est = y0;
	elseif abs(corr_x)<0.5001
		x0_est = x0 + corr_x*dx;
		corr_y = (yp-ym)^2/8/(2*y0-ym-yp);
		y0_est = y0 + corr_y;
	else
		keyboard
	end
end

function initSystemInfo = initSystemInfo(mib,sfn4lsb,k_SSB,L_max)
	% Create set of subcarrier spacings signaled by the 7th bit of the
	% decoded MIB, the set is different for FR1 (L_max=4 or 8) and FR2
	% (L_max=64)
	if (L_max==64)
		scsCommon = [60 120];
	else
		scsCommon = [15 30];
	end
	initSystemInfo = struct();
	initSystemInfo.NFrame = mib.systemFrameNumber*2^4 + bit2int(sfn4lsb,4);
	initSystemInfo.SubcarrierSpacingCommon = scsCommon(mib.subCarrierSpacingCommon + 1);
	initSystemInfo.k_SSB = k_SSB + mib.ssb_SubcarrierOffset;
	initSystemInfo.DMRSTypeAPosition = 2 + mib.dmrs_TypeA_Position;
	initSystemInfo.PDCCHConfigSIB1 = info(mib.pdcch_ConfigSIB1);
	initSystemInfo.CellBarred = mib.cellBarred;
	initSystemInfo.IntraFreqReselection = mib.intraFreqReselection;
end