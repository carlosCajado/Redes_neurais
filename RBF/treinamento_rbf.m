% Carregar os dados DIREITA!!
load('dados_audio_DI.mat');
load('dados_audio_REI.mat');
load('dados_audio_TA.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES.mat');
load('dados_audio_QUER.mat');
load('dados_audio_DA.mat');

% Número de neuronios na camada de saída
numClasses = 6;         

% Número de neurônios na camada oculta
numRBFNeurons = 12;      

% Parâmetros da rede RBF
eta = 0.001;          % Taxa de aprendizado
maxEpocas = 5000;    % Número máximo de épocas para o treinamento

% Concatenar os dados
dados_di = cat(1, dados_audio_di{:});
dados_rei = cat(1, dados_audio_rei{:});
dados_ta = cat(1, dados_audio_ta{:});
dados_es = cat(1, dados_audio_es{:});
dados_quer = cat(1, dados_audio_quer{:});
dados_da = cat(1, dados_audio_da{:});

% 6 CLASSES; DI, REI, TA, ES, QUER, DA
X = [dados_di; dados_rei; dados_ta; dados_es;  dados_quer; dados_da];

Y = [1*ones(size(dados_di', 2), 1); 2*ones(size(dados_rei', 2), 1); 3*ones(size(dados_ta', 2), 1);
        4*ones(size(dados_es', 2), 1); 5*ones(size(dados_quer', 2), 1); 6*ones(size(dados_quer', 2), 1)];

% Normalização para [0,1]
X = (X - min(X(:))) / (max(X(:)) - min(X(:)));

% Projeto da camada escondida: determinar centros dos neurônios usando WTA
centers = X(randperm(size(X, 1), numRBFNeurons), :);

% Determinação da abertura dos neurônios (sigma)
sigma = zeros(numRBFNeurons, 1);
for i = 1:numRBFNeurons
    totalDist = 0;
    count = 0;
    for j = 1:numRBFNeurons
        if i ~= j
            dist = sqrt(sum((centers(i, :) - centers(j, :)).^2));
            totalDist = totalDist + dist;
            count = count + 1;
        end
    end
    sigma(i) = totalDist / count;
end

% Projeto da camada de saída: inicializar pesos e bias
Woh = rand(numClasses, numRBFNeurons);
bias_oh = rand(numClasses, 1);

% Constante k para calcular a saída da rede
k = 1;
erro_epoca = [];

% Loop de treinamento
for epoca = 1:maxEpocas
    % Calcular a saída da camada RBF

    Yh = zeros(size(X, 1), numRBFNeurons);
    for i = 1:numRBFNeurons
        mu_i = sqrt(sum((X - centers(i, :)).^2, 2)); % Calculo de mu_i
        Yh(:, i) = exp(-mu_i.^2 / (2 * sigma(i)^2)); % Função Gaussiana
    end
    %  Calcular entrada da camada de saída
    
    net_o = Woh * Yh' + bias_oh * ones(1, size(Yh', 2));

    % Calcular a saída da rede neural
    Ys = k * net_o';
    % Calcular erro
    E = Y - Ys;
    df = ones(size(net_o));

    % Calcular novos valores dos pesos
    delta_bias_oh = eta * sum((E'.* df)')';
    delta_Woh = eta *(E'.* df) * Yh;

    % Atualizar pesos e bias
    Woh = Woh + delta_Woh;
    bias_oh = bias_oh + delta_bias_oh;
    
    % Calcular erro quadrático médio Eav da rede
    Eav = mean(E.^2);
    
    % Coleta o erro a cada 10 épocas.
    if mod(epoca, 10) == 0
        erro_epoca(end+1) = sum(Eav);
    end
end

% Salvar os pesos e bias treinados
save('pesos_bias_treinados_RBF_WTA.mat', 'centers', 'sigma', 'Woh', 'bias_oh', 'numRBFNeurons');

num_epocas = 1:10:maxEpocas;
% Plotar o gráfico
fig = figure;
plot(num_epocas, erro_epoca, 'b-', 'LineWidth', 2);
xlabel('Época');
ylabel('Erro');
title('Erro por Época');
grid on;

% Salvar o gráfico
savefig(fig, 'erro_epoca_RBF_WTA.fig');
