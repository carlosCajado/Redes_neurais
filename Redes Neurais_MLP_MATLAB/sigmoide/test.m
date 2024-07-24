
% Carregar os dados de teste
dados_teste = readtable('car_teste.csv');

% Separar características (features) das classes
XDados_teste  = dados_teste(:, 1:end-4);
YDados_teste  = dados_teste(:, end-3:end);

X_teste = table2array(XDados_teste);  % Características
Y_teste =  table2array(YDados_teste);      % Classes

% Inicializar variáveis para armazenar resultados
num_amostras = size(X_teste, 1);
predicoes = zeros(num_amostras, 1);

% Loop de teste
for i = 1:num_amostras
    
    % Passo 2: Calcular entrada da camada escondida
    net_h = Whi * X_teste(i, :)' + bias_hi;

    % Passo 3: Calcular a saída da camada escondida
    Yh = logsig(net_h);

    % Passo 4: Calcular entrada da camada de saída
    net_o = Woh * Yh + bias_oh;
    
    % Passo 5: Calcular a saída da rede neural
    Ys = k * net_o; % linear
    disp(max(Ys))
    % Armazenar a classe prevista
    [~, predicao] = max(Ys);

    predicoes(i) = predicao;
    
end
% Calcular a acurácia
acuracia = sum(predicoes == Y_teste) / num_amostras;
disp(['A acurácia da rede neural no conjunto de teste é: ', num2str(acuracia * 100), '%']);


% Definindo a ordem das classes
classes_ordem = {'unacc', 'acc', 'good', 'vgood'};

% Contando o número de valores em cada classe
n_valores_classe = zeros(1, length(classes_ordem));

for i = 1:length(classes_ordem)
    % Obtendo o índice da coluna correspondente à classe
    indice_classe = find(strcmp(classes_ordem{i}, {'unacc', 'acc', 'good', 'vgood'}));
    
    % Contando o número de ocorrências da classe na coluna
    n_valores_classe(i) = sum(Y_teste(:, indice_classe));
end

% Exibindo o número de valores em cada classe
for i = 1:length(classes_ordem)
    fprintf("Classe %s: %d\n", classes_ordem{i}, n_valores_classe(i));
end
