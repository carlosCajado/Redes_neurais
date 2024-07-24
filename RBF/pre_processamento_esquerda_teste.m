% Diretório para cada sigla
diretorio_ES    = 'teste\esquerda_silabas\ES';
diretorio_QUER  = 'teste\esquerda_silabas\QUER';
diretorio_DA    = 'teste\esquerda_silabas\DA';

% Carregando audios 
audio_es = dir(fullfile(diretorio_ES, '*.wav')); 
audio_quer = dir(fullfile(diretorio_QUER, '*.wav')); 
audio_da = dir(fullfile(diretorio_DA, '*.wav')); 

dados_audio_es_teste = cell(1, numel(audio_es));
dados_audio_quer_teste = cell(1, numel(audio_quer));
dados_audio_da_teste = cell(1, numel(audio_da));

% Número de Partes
N_partes = 128;

% áudios ES
for i = 1:numel(audio_es)
     arquivo_atual = fullfile(diretorio_ES, audio_es(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
      fft_data = abs(fft(audio_data(:,1)));

     % Dividindo fft_data em partes 
     tamanho_parte = floor(numel(fft_data) / N_partes);
     partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);
     dados_audio_es_teste{i} =  mean(partes_fft);

end

% áudios QUER
for i = 1:numel(audio_quer)
     arquivo_atual = fullfile(diretorio_QUER, audio_quer(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
     fft_data = abs(fft(audio_data(:,1)));

     % Dividindo fft_data em partes 
     tamanho_parte = floor(numel(fft_data) / N_partes);
     partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);
     dados_audio_quer_teste{i} =  mean(partes_fft);

end

%  áudios DA
for i = 1:numel(audio_da)
     arquivo_atual = fullfile(diretorio_DA, audio_da(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
      fft_data = abs(fft(audio_data(:,1)));

     % Dividindo fft_data em partes 
     tamanho_parte = floor(numel(fft_data) / N_partes);
     partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);

     dados_audio_da_teste{i} =  mean(partes_fft);
end

% Salvando...
save('dados_audio_ES_teste.mat', 'dados_audio_es_teste');
save('dados_audio_QUER_teste.mat', 'dados_audio_quer_teste');
save('dados_audio_DA_teste.mat', 'dados_audio_da_teste');

