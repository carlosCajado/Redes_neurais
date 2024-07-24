% Carregar os dados DIREITA!!
load('dados_audio_DI_teste.mat');
load('dados_audio_REI.mat');
load('dados_audio_TA.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES.mat');
load('dados_audio_QUER.mat');
load('dados_audio_DA.mat');

load('pesos_bias_treinados.mat')

% Atribuir os dados es a X
X = cat(1, dados_audio_quer_teste{:});
k = 1;

% Passo 2: Calcular entrada da camada escondida
net_h_di = Whi * X' + bias_hi * ones(1, size(X', 2));

% Passo 3: Calcular a saída da camada escondida
Yh_di = logsig(net_h_di);

% Passo 4: Calcular entrada da camada de saída
net_o_di = Woh * Yh_di + bias_oh * ones(1, size(Yh_di, 2));

% Passo 5: Calcular a saída da rede neural para os dados DI
Ys_di =k * net_o_di;

classe_prevista = round(min(Ys_di,[],1));
disp(classe_prevista)
