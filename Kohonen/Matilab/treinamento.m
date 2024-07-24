% Carregar os dados DIREITA!!
load('dados_audio_DI.mat');
load('dados_audio_REI.mat');
load('dados_audio_TA.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES.mat');
load('dados_audio_QUER.mat');
load('dados_audio_DA.mat');


dados_di = cat(1, dados_audio_di{:});
dados_rei= cat(1, dados_audio_rei{:});
dados_ta = cat(1, dados_audio_ta{:});

dados_es = cat(1, dados_audio_es{:});
dados_quer = cat(1, dados_audio_quer{:});
dados_da = cat(1, dados_audio_da{:});


% CLASSES; DI, REI, TA, ES, QUER, DA
X = [dados_di; dados_rei; dados_ta; dados_es;  dados_quer; dados_da];

Y = [1*ones(size(dados_di', 2), 1); 2*ones(size(dados_rei', 2), 1); 3*ones(size(dados_ta', 2), 1);
        4*ones(size(dados_es', 2), 1); 5*ones(size(dados_quer', 2), 1); 6*ones(size(dados_quer', 2), 1)];

% Parâmetros da rede neural

sigma0 = 2;         % Sigma inicial
eta0 = 0.01;        % Taxa de aprendizado inicial
tau_sigma = 10000;  % Tempo de decaimento do sigma
tau_eta = 12000;    % Tempo de decaimento da taxa de aprendizado
epocas = 10000;     % Número de épocas
Neuronios = 24;     % Número de neurônios


% Inicialização aleatória dos pesos
pesos = rand(size(X, 2), Neuronios);

% Outros Paramentros;
max_delta_w_significativo = 1e-6;

% Função de vizinhança, Processo cooperativo:
funcao_vizinhanca = @(distancia, sigma) exp(-distancia^2 / (2 * sigma^2));

% Treinamento
eta_atual   = eta0;
sigma_atual = sigma0;

for t = 1:epocas
    for a = 1:size(X, 1)

        %Processo competitivo:
        amostra = X(a, :)';
        
        % Encontrar o neurônio vencedor
        %distância euclidiana entre o vetor x e os vetores de pesos 

        distancias = sum((pesos - amostra).^2, 1);
        [~, vencedor] = min(distancias);
        
        % Iniciando os Pesos
        pesosAnteriores = pesos;

        %Para cada neurônio de saída temos: 
        for n = 1:Neuronios
            %Processo cooperativo:
            distancia = (n - vencedor)^2;
            h = funcao_vizinhanca(distancia, sigma_atual);

            %Processo adaptativo (ajustar os valores dos pesos):
            pesos(:, n) = pesos(:, n) + eta_atual * h * (amostra - pesos(:, n));
        end
    end

    % Atualizar eta e sigma
    eta_atual = eta0 * exp(-t / tau_eta);
    sigma_atual = sigma0 * exp(-t / tau_sigma);

    % Impressão de progresso
    max_delta_w = max(max(abs(pesos - pesosAnteriores)));
    if mod(t, 100) == 0
        disp(['Época ', num2str(t), ': max_delta_w = ', num2str(max_delta_w)]);
    end

    % Critério de parada
    if max_delta_w < max_delta_w_significativo
        disp(['Critério de parada atingido na época ', num2str(t)]);
        break; 
    end
end

% Determinação dos Vencedores
numAmostras = size(X, 1);
vencedores = zeros(numAmostras, 1);
for amostra = 1:numAmostras
    [~, vencedores(amostra)] = min(sum((pesos - X(amostra, :)').^2, 1));
end

% Atribuição de Classes
classificacao = zeros(Neuronios, 1);
for neuronio = 1:Neuronios
    amostras = find(vencedores == neuronio);
    if ~isempty(amostras)
        classificacao(neuronio) = mode(Y(amostras));
    end
end

save('pesos_bias_treinados.mat', 'pesos');
