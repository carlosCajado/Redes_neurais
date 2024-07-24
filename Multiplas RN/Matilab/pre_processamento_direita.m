% Diretório para cada sigla
diretorio_DI = 'treino\Direita-silabas\DI';
diretorio_REI = 'treino\Direita-silabas\REI';
diretorio_TA = 'treino\Direita-silabas\TA';

% Carregando audios e inicializando células para armazenar dados de FFT
audio_di = dir(fullfile(diretorio_DI, '*.wav')); 
audio_rei = dir(fullfile(diretorio_REI, '*.wav')); 
audio_ta = dir(fullfile(diretorio_TA, '*.wav')); 


dados_audio_di = cell(1, numel(audio_di));
dados_audio_rei = cell(1, numel(audio_rei));
dados_audio_ta = cell(1, numel(audio_ta));

%conjunto de partes de amostras 
N_partes = 128;

% áudios DI
for i = 1:numel(audio_di)

     arquivo_atual = fullfile(diretorio_DI, audio_di(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
     fft_data = abs(fft(audio_data(:,1)));

    % Dividindo fft_data em partes 
    tamanho_parte = floor(numel(fft_data) / N_partes);
    partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);
    
    % Calculando a média de cada parte
     dados_audio_di{i} = mean(partes_fft) ;
end

% áudios REI
for i = 1:numel(audio_rei)
     arquivo_atual = fullfile(diretorio_REI, audio_rei(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
     fft_data = abs(fft(audio_data(:,1)));

     % Dividindo fft_data em partes 
     tamanho_parte = floor(numel(fft_data) / N_partes);
     partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);
     dados_audio_rei{i} =  mean(partes_fft);

end

%  áudios TA
for i = 1:numel(audio_ta)
     arquivo_atual = fullfile(diretorio_TA, audio_ta(i).name);
     [audio_data, ~] = audioread(arquivo_atual);
     fft_data = abs(fft(audio_data(:,1)));

     % Dividindo fft_data em partes 
     tamanho_parte = floor(numel(fft_data) / N_partes);
     partes_fft = reshape(fft_data(1:N_partes*tamanho_parte), tamanho_parte, N_partes);
     dados_audio_ta{i} =  mean(partes_fft);
end

% Salvando...
save('dados_audio_DI.mat', 'dados_audio_di');
save('dados_audio_REI.mat', 'dados_audio_rei');
save('dados_audio_TA.mat', 'dados_audio_ta');
