
% Parâmetros da rede neural
numClasses = 4;         % Número de classes na camada de saída
hiddenLayerSize = 2;    % Número de neurônios na camada oculta

% Passo 1: Ler os dados do arquivo .data
dados_validacao = readtable('car_validacao.csv');

% Separar características (features) das classes
XDados_validacao  = dados_validacao(:, 1:end-4);
YDados_validacao  = dados_validacao(:, end-3:end);

X_validacao = table2array(XDados_validacao);  % Características
Y_validacao =  table2array(YDados_validacao);      % Classes

% Carregar pesos e bias previamente treinados
load('pesos_bias_treinados.mat');

% Calcular saída da camada escondida para dados de validação
net_h_validacao = Whi * X_validacao' + bias_hi * ones(1, size(X_validacao', 2));
Yh_validacao = tanh(net_h_validacao);

% Calcular saída da camada de saída para dados de validação
net_o_validacao = Woh * Yh_validacao + bias_oh * ones(1, size(Yh_validacao, 2));
Ys_validacao = net_o_validacao;

% Calcular acurácia da validação
[~, Y_pred_validacao] = max(Ys_validacao);
[~, Y_true_validacao] = max(Y_validacao');
acuracia_validacao = sum(Y_pred_validacao == Y_true_validacao) / length(Y_true_validacao);

% Exibir acurácia da validação
fprintf('Acurácia da validação: %.2f%%\n', acuracia_validacao * 100);
