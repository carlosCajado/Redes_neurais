% Carregar os dados de teste
load('dados_audio_DI_teste.mat');
load('dados_audio_REI_teste.mat');
load('dados_audio_TA_teste.mat');
load('dados_audio_ES_teste.mat');
load('dados_audio_QUER_teste.mat');
load('dados_audio_DA_teste.mat');

% Carregar os pesos e bias treinados
load('pesos_bias_treinados_RBF_WTA.mat');

% Concatenar os dados de teste
dados_teste_di = cat(1, dados_audio_di_teste{:});
dados_teste_rei = cat(1, dados_audio_rei_teste{:});
dados_teste_ta = cat(1, dados_audio_ta_teste{:});
dados_teste_es = cat(1, dados_audio_es_teste{:});
dados_teste_quer = cat(1, dados_audio_quer_teste{:});
dados_teste_da = cat(1, dados_audio_da_teste{:});

% 6 CLASSES; DI, REI, TA, ES, QUER, DA
X_teste = [dados_teste_di; dados_teste_rei; dados_teste_ta; dados_teste_es; dados_teste_quer; dados_teste_da];
Y_teste = [1*ones(size(dados_teste_di', 2), 1); 2*ones(size(dados_teste_rei', 2), 1); 3*ones(size(dados_teste_ta', 2), 1);
           4*ones(size(dados_teste_es', 2), 1); 5*ones(size(dados_teste_quer', 2), 1); 6*ones(size(dados_teste_da', 2), 1)];

% Normalização para [0,1] 
X_teste = (X_teste - min(X_teste(:))) / (max(X_teste(:)) - min(X_teste(:)));

% Calculando a saída da camada RBF
Yh_test = zeros(size(X_teste, 1), numRBFNeurons);
for i = 1:numRBFNeurons
    mu_i = sqrt(sum((X_teste - centers(i, :)).^2, 2)); % Calculo de mu_i
    Yh_test(:, i) = exp(-mu_i.^2 / (2 * sigma(i)^2)); % Função Gaussiana
end

% 1 - Calcular a entrada da camada de saída
net_o_test = Woh * Yh_test' + bias_oh * ones(1, size(Yh_test', 2));

% 2 - Calcular a saída da rede neural
Ys_test = k * net_o_test';

% 3 - Classificar as saídas (classe com a maior probabilidade)
[~, Y_pred] = max(Ys_test, [], 2);
disp(Y_pred)
% 4 - Calcular a acurácia do teste
accuracy = sum(Y_pred == Y_teste) / length(Y_teste);

fprintf('Acurácia do teste: %.2f%%\n', accuracy * 100);

conf_matrix = confusionmat(Y_teste, Y_pred);

figure;
confusionchart(conf_matrix);
title('Matriz de Confusão para Dados de Teste');
xlabel('Classe Predita');
ylabel('Classe Verdadeira');

% Calcular acertos
acertos = sum(Y_pred == Y_teste);
acuracia = acertos / length(Y_teste) * 100;
fprintf('Acurácia: %.2f%%\n', acuracia);
