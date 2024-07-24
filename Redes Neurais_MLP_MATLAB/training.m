clear
clc

% Parâmetros da rede neural
numClasses = 4;         % Número de classes na camada de saída
hiddenLayerSize = 2;    % Número de neurônios na camada oculta
eta = 0.01;            % Taxa de aprendizado
%sigmoid = @(x) 1 ./ (1 + exp(-x));

% Passo 1: Ler os dados do arquivo .data
dados = readtable('car.csv');

% Separar características (features) das classes
XDados  = dados(:, 1:end-4);
YDados  = dados(:, end-3:end);

X = table2array(XDados);  % Características
Y =  table2array(YDados);      % Classes

% Inicializar pesos e bias
Whi = rand(hiddenLayerSize, size(X, 2)) - 0.5;
bias_hi = rand(hiddenLayerSize, 1) - 0.5;
Woh = rand(numClasses, hiddenLayerSize) - 0.5;
bias_oh = rand(numClasses, 1) - 0.5;

% Constante k para calcular a saída da rede
k = 1;

% Definir o número máximo de épocas
maxEpocas = 110;

erro_epoca = [];

% Loop de treinamento
for epoca = 1:maxEpocas
    
    % Passo 2: Calcular entrada da camada escondida
    net_h = Whi * X' + bias_hi * ones(1, size(X', 2));

    % Passo 3: Calcular a saída da camada escondida
    Yh = logsig(net_h);

    % Passo 4: Calcular entrada da camada de saída
    net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2));
    
    % Passo 5: Calcular a saída da rede neural
    Ys = k * net_o; 
    
    % Passo 6: Calcular erro retropropagado
    E = Y' - Ys; %  slide e Y, no lugar da transposta
    
    % Passo 7: Calcular novos valores dos pesos
    df = ones(size(net_o));
    
    % Calcular a variação  da camada de saída
    delta_bias_oh = eta * sum((E.* df)')';
    % Calcular a variação dos pesos entre as camadas de saída e escondida
    delta_Woh = eta * (E.* df) * Yh';
    
    % 8. Calcular erro retropropagado
    Eh = -Woh' * (E.* df);
    
    % 9. Calcular variação dos pesos entre as camadas escondida e de entrada
    df = logsig(net_h) - (logsig(net_h).^2);
    delta_bias_hi = -eta * sum((Eh.* df)')';
    delta_Whi = -eta * (Eh.* df) * X;% delta_Whi = -eta * (Eh.* df) * X';

   % Passo 10: Calcular novos valores dos pesos
    Whi = Whi + delta_Whi;
    Woh = Woh + delta_Woh;
    bias_hi = bias_hi + delta_bias_hi;
    bias_oh = bias_oh + delta_bias_oh;

    % Passo 11: Calcular erro quadrático médio Eav da rede
    Eav = mean(E.^2);
    % Coletar o erro a cada 10 épocas
    if mod(epoca, 10) == 0
        erro_epoca = [epoca, Eav];
            disp(Eav)
    end
    
end

% Plotar o erro quadrático médio em relação ao número de épocas
figure;
plot(erro_epoca,1:length(erro_epoca), 'b-', 'LineWidth', 2);
xlabel('Época');
ylabel('Erro Quadrático Médio');
title('Convergência do Erro');
grid on;
