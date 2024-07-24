% Carregar os dados DIREITA!!
load('dados_audio_DI_teste.mat');
load('dados_audio_REI_teste.mat');
load('dados_audio_TA_teste.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES_teste.mat');
load('dados_audio_QUER_teste.mat');
load('dados_audio_DA_teste.mat');

load('pesos_bias_treinados.mat');

dados_teste_di = cat(1, dados_audio_di_teste{:});
dados_teste_rei= cat(1, dados_audio_rei_teste{:});
dados_teste_ta = cat(1, dados_audio_ta_teste{:});

dados_teste_es = cat(1, dados_audio_es_teste{:});
dados_teste_quer = cat(1, dados_audio_quer_teste{:});
dados_teste_da = cat(1, dados_audio_da_teste{:});

%Temos 6 CLASSES; DI, REI, TA, ES, QUER, DA
X_teste = [dados_teste_di; dados_teste_rei; dados_teste_ta; dados_teste_es;  dados_teste_quer; dados_teste_da];

Y_teste = [1*ones(size(dados_teste_di', 2), 1); 2*ones(size(dados_teste_rei', 2), 1); 3*ones(size(dados_teste_ta', 2), 1);
        4*ones(size(dados_teste_es', 2), 1); 5*ones(size(dados_teste_quer', 2), 1); 6*ones(size(dados_teste_da', 2), 1)];

% Passo 2: Calcular entrada da camada escondida
net_h_teste = Whi * X_teste' + bias_hi * ones(1, size(X_teste', 2));

% Passo 3: Calcular a saída da camada escondida
Yh_teste = logsig(net_h_teste);

% Passo 4: Calcular entrada da camada de saída
net_o_teste = Woh * Yh_teste + bias_oh * ones(1, size(Yh_teste, 2));

% Passo 5: Calcular a saída da rede neural
Ys_teste = net_o_teste; 

% calcular a classe prevista para cada amostra.
Ys_mean = mean(Ys_teste);
classe_prevista = round(max(Ys_mean,[],1));
acertos = 0;

for i = 1:length(classe_prevista)
    fprintf('Amostra %d: Valor Previsto = %d, Valor Real = %d\n', i, classe_prevista(i), Y_teste(i));
    if classe_prevista(i) == Y_teste(i)
        acertos = acertos + 1;
    end
end

% calcular a acurácia.
acuracia = acertos / length(classe_prevista) * 100;
fprintf('Acurácia: %.2f%%\n', acuracia);

