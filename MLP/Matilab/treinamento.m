% Carregar os dados DIREITA!!
load('dados_audio_DI.mat');
load('dados_audio_REI.mat');
load('dados_audio_TA.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES.mat');
load('dados_audio_QUER.mat');
load('dados_audio_DA.mat');

% Parâmetros da rede neural
numClasses = 6;         % Número de classes na camada de saída
hiddenLayerSize = 4;    % Número de neurônios na camada oculta
eta = 0.0001;            % Taxa de aprendizado

% Definir o número máximo de épocas
maxEpocas = 50000;

dados_di = cat(1, dados_audio_di{:});
dados_rei= cat(1, dados_audio_rei{:});
dados_ta = cat(1, dados_audio_ta{:});

dados_es = cat(1, dados_audio_es{:});
dados_quer = cat(1, dados_audio_quer{:});
dados_da = cat(1, dados_audio_da{:});


%6 CLASSES; DI, REI, TA, ES, QUER, DA
X = [dados_di; dados_rei; dados_ta; dados_es;  dados_quer; dados_da];

Y = [1*ones(size(dados_di', 2), 1); 2*ones(size(dados_rei', 2), 1); 3*ones(size(dados_ta', 2), 1);
        4*ones(size(dados_es', 2), 1); 5*ones(size(dados_quer', 2), 1); 6*ones(size(dados_quer', 2), 1)];

% % Passo 1: Inicializar pesos e bias aleatoriamente
Whi = rand(hiddenLayerSize, size(X, 2)) - 0.5;

bias_hi = rand(hiddenLayerSize, 1) - 0.5;
%bias_hi = 1;
Woh = rand(numClasses, hiddenLayerSize) - 0.5;
bias_oh = rand(numClasses, 1) - 0.5;
%bias_oh = 1;

% Constante k para calcular a saída da rede
k = 1;
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
    E = Y - Ys'; %  slide e Y, no lugar da transposta

    % Passo 7: Calcular novos valores dos pesos
    df = ones(size(net_o));

    % Calcular a variação  da camada de saída
    delta_bias_oh = eta * sum((E'.* df)')';

    % Calcular a variação dos pesos entre as camadas de saída e escondida
    delta_Woh = eta *(E'.* df) * Yh';

    % 8. Calcular erro retropropagado
    Eh = -Woh' * (E'.* df);
    
    % 9. Calcular variação dos pesos entre as camadas escondida e de entrada
    df = logsig(net_h) - (logsig(net_h).^2);
    delta_bias_hi = -eta * sum((Eh.* df)')';
    delta_Whi = -eta * (Eh.* df) * X;

   % Passo 10: Calcular novos valores dos pesos
    Whi = Whi + delta_Whi;
    Woh = Woh + delta_Woh;
    bias_hi = bias_hi + delta_bias_hi;
    bias_oh = bias_oh + delta_bias_oh;

    % Passo 11: Calcular erro quadrático médio Eav da rede
    Eav = mean(E.^2);
    
    % Coleta o erro a cada 100 épocas.
    if mod(epoca, 1000) == 0
        erro_epoca(epoca) = sum(Eav);
        disp(Eav)
    end
    
end

%salva
save('pesos_bias_treinados.mat', 'Whi', 'Woh', 'bias_hi', 'bias_oh');

num_epocas = 1:numel(erro_epoca);
% Plotar o gráfico
fig = figure;
plot(num_epocas, erro_epoca, 'b-', 'LineWidth', 2);
xlabel('Época');
ylabel('Erro');
title('Erro por Época');
grid on;

%salva o gráfico
savefig(fig, 'erro_epoca.fig');

