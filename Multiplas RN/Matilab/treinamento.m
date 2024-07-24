% Carregar os dados DIREITA!!
load('dados_audio_DI.mat');
load('dados_audio_REI.mat');
load('dados_audio_TA.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES.mat');
load('dados_audio_QUER.mat');
load('dados_audio_DA.mat');

% Parâmetros da rede neural

numClasses = 6;      % Número de classes na camada de saída
hiddenLayerSize = 6; % Número de neurônios na camada oculta
eta = 0.001;        % Taxa de aprendizado
maxEpocas = 10000;  % Definir o número máximo de épocas
NRedes = 3;          % Definir o número de Redes Neurais MLP

dados_di = cat(1, dados_audio_di{:});
dados_rei= cat(1, dados_audio_rei{:});
dados_ta = cat(1, dados_audio_ta{:});

dados_es = cat(1, dados_audio_es{:});
dados_quer = cat(1, dados_audio_quer{:});
dados_da = cat(1, dados_audio_da{:});

% 6 CLASSES; DI, REI, TA, ES, QUER, DA
X = [dados_di; dados_rei; dados_ta; dados_es; dados_quer; dados_da];
Y = [1*ones(size(dados_di, 1), 1); 2*ones(size(dados_rei, 1), 1); 3*ones(size(dados_ta, 1), 1);
     4*ones(size(dados_es, 1), 1); 5*ones(size(dados_quer, 1), 1); 6*ones(size(dados_da, 1), 1)];

% Tratando o Y para a forma de vetores binários.
Y_onehot = full(ind2vec(Y'))';

% Inicializar variáveis para armazenar pesos, biases e erros
Whi_all = cell(1, NRedes);
Woh_all = cell(1, NRedes);
bias_hi_all = cell(1, NRedes);
bias_oh_all = cell(1, NRedes);
erro_epoca_all = nan(maxEpocas, NRedes);

E = Y_onehot; % Inicializar E como Y_onehot para a primeira rede..

% Treinar as redes neurais
for rede = 1:NRedes
    [Whi, Woh, bias_hi, bias_oh, erro_epoca, Ys] = treinarRede(X, E, hiddenLayerSize, numClasses, eta, maxEpocas, rede);
    Whi_all{rede} = Whi;
    Woh_all{rede} = Woh;
    bias_hi_all{rede} = bias_hi;
    bias_oh_all{rede} = bias_oh;
    erro_epoca_all(:, rede) = erro_epoca;
    
    % Atualiza o erro para a próxima rede...
    E = E - Ys';
    
    disp(['Rede ', num2str(rede)]);

    % diminuindo o eta... 
    if rede >= 2
        eta = (eta/10); 
    end
end

% Salvar os pesos e bias treinados
save('pesos_bias_treinados.mat', 'Whi_all', 'Woh_all', 'bias_hi_all', 'bias_oh_all', 'NRedes');

% Calcular a média do erro por época para as redes
erro_epoca_combined = mean(erro_epoca_all, 2, 'omitnan');

% Plot
fig = figure;
plot(1:maxEpocas, erro_epoca_combined, 'b-', 'LineWidth', 2);
xlabel('Época');
ylabel('Erro');
title('Erro por Época');
grid on;

savefig(fig, 'erro_epoca_combined.fig');

% treina a rede neural
function [Whi, Woh, bias_hi, bias_oh, erro_epoca, Ys] = treinarRede(X, Y, hiddenLayerSize, numClasses, eta, maxEpocas, rede)
    % Passo 1: Inicializando os pesos e bias
    Whi = rand(hiddenLayerSize, size(X, 2)) - 0.5;
    bias_hi = rand(hiddenLayerSize, 1) - 0.5;
    Woh = rand(numClasses, hiddenLayerSize);
    bias_oh = rand(numClasses, 1);

    % Constante k para calcular a saída da rede -- padrão = 1, (constante)
    k = 1;
    erro_epoca = nan(maxEpocas, 1);
    
    % Loop de treinamento
    for epoca = 1:maxEpocas

       % Zerando a saida da rede para o incio da segunda
       if rede >= 2 && epoca == 1
            Woh = 0;
            bias_oh = 0;
       end

        % Passo 2: Calcular entrada da camada escondida
        net_h = Whi * X' + bias_hi * ones(1, size(X', 2));
        
        % Passo 3: Calcular a saída da camada escondida
        Yh = logsig(net_h);
        
        % Passo 4: Calcular entrada da camada de saída
        net_o = Woh * Yh + bias_oh * ones(1, size(Yh, 2));

        % Passo 5: Calcular a saída da rede neural
        Ys = logsig(net_o);
        
        % Zerando a saida da rede para o incio da segunda
        if rede >= 2 && epoca == 1
            Ys = 0;
        end

        % Passo 6: Calcular erro retropropagado
        E = Y' - Ys;
        
        % Passo 7: Calcular novos valores dos pesos
        df = Ys .* (1 - Ys);
        
            % Calcular a variação da camada de saída
        delta_bias_oh = eta * sum((E .* df), 2);
        
            % Calcular a variação dos pesos entre as camadas de saída e escondida
        delta_Woh = eta * (E .* df) * Yh';
        
        % Passo :8 Calcular erro retropropagado
        Eh = Woh' * (E .* df);
        
        % Passo :9 Calcular variação dos pesos entre as camadas escondida e de entrada
        df_hidden = Yh .* (1 - Yh);
        delta_bias_hi = eta * sum((Eh .* df_hidden), 2);
        delta_Whi = eta * (Eh .* df_hidden) * X;
        
        % Passo 10: Calcular novos valores dos pesos
        Whi = Whi + delta_Whi;
        Woh = Woh + delta_Woh;
        bias_hi = bias_hi + delta_bias_hi;
        bias_oh = bias_oh + delta_bias_oh;

        % Passo 11: Calcular erro quadrático médio Eav da rede
        Eav = mean(E(:).^2);
        
        % Coleta o erro a x época.
        erro_epoca(epoca) = Eav;
        if mod(epoca, 100) == 0
            disp(['Época ', num2str(epoca), ' Erro: ', num2str(Eav)]);
        end
    end
end