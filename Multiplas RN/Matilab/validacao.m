% Carregar os dados DIREITA!!
load('dados_audio_DI_validacao.mat');
load('dados_audio_REI_validacao.mat');
load('dados_audio_TA_validacao.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES_validacao.mat');
load('dados_audio_QUER_validacao.mat');
load('dados_audio_DA_validacao.mat');

load('pesos_bias_treinados.mat');

dados_validacao_di = cat(1, dados_audio_di_validacao{:});
dados_validacao_rei= cat(1, dados_audio_rei_validacao{:});
dados_validacao_ta = cat(1, dados_audio_ta_validacao{:});

dados_validacao_es = cat(1, dados_audio_es_validacao{:});
dados_validacao_quer = cat(1, dados_audio_quer_validacao{:});
dados_validacao_da = cat(1, dados_audio_da_validacao{:});

%Temos 6 CLASSES; DI, REI, TA, ES, QUER, DA
X_validacao = [dados_validacao_di; dados_validacao_rei; dados_validacao_ta; dados_validacao_es;  dados_validacao_quer; dados_validacao_da];

Y_validacao = [1*ones(size(dados_validacao_di', 2), 1); 2*ones(size(dados_validacao_rei', 2), 1); 3*ones(size(dados_validacao_ta', 2), 1);
        4*ones(size(dados_validacao_es', 2), 1); 5*ones(size(dados_validacao_quer', 2), 1); 6*ones(size(dados_validacao_da', 2), 1)];


Ys_validacao_final = zeros(size(Y_validacao'));

for rede = 1:NRedes
    % Obter pesos e biases da rede atual
    Whi = Whi_all{rede};
    Woh = Woh_all{rede};
    bias_hi = bias_hi_all{rede};
    bias_oh = bias_oh_all{rede};
    
    % Passo 2: Calcular entrada da camada escondida
    net_h_validacao = Whi * X_validacao' + bias_hi * ones(1, size(X_validacao', 2));

    % Passo 3: Calcular a saída da camada escondida
    Yh_validacao = logsig(net_h_validacao);

    % Passo 4: Calcular entrada da camada de saída
    net_o_validacao = Woh * Yh_validacao + bias_oh * ones(1, size(Yh_validacao, 2));

    % Passo 5: Calcular a saída da rede neural
    Ys_validacao = logsig(net_o_validacao);

    % Atualizar a saída final acumulada
    Ys_validacao_final = Ys_validacao_final + Ys_validacao;

    
end

% Calcular a classe prevista para cada amostra
[~, classe_prevista] = max(Ys_validacao_final, [], 1)
acertos = 0;


% Calcular a matriz de confusão
conf_matrix = confusionmat(Y_validacao, classe_prevista');

% Plotar a matriz de confusão
fig = figure;
bar3(conf_matrix);

title('Matriz de Confusão');
xlabel('Classe Prevista');
ylabel('Classe Verdadeira');
zlabel('Contagem');
savefig(fig, 'conf_matrix.fig');