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

numAmostrasValidacao = size(X_validacao, 1);
numNeuronios = size(pesos, 2);
vencedores = zeros(numAmostrasValidacao, 1);

% Encontrar os neurônios vencedores para cada amostra de validacao.
for i = 1:numAmostrasValidacao
    distancias = sum((pesos - X_validacao(i, :)').^2, 1);
    [~, vencedor] = min(distancias);
    vencedores(i) = vencedor;
end

rotulosvalidacao = Y_validacao;
numClasses = length(unique(rotulosvalidacao));
rotuloMaisComum = zeros(numNeuronios, 1);

% Encontrar o rótulo mais comum para cada neurônio, temos:
for neuronio = 1:numNeuronios
    indicesNeuronio = find(vencedores == neuronio);
    classesNeuronio = rotulosvalidacao(indicesNeuronio);
    if ~isempty(classesNeuronio)
        rotuloMaisComum(neuronio) = mode(classesNeuronio);
    end
end

numAcertos = 0;
classePrevista = zeros(numAmostrasValidacao, 1);

% Prever a classe para cada amostra de validacao
for i = 1:numAmostrasValidacao
    neuronio = vencedores(i);
    classePrevista(i) = rotuloMaisComum(neuronio);
    if rotulosvalidacao(i) == classePrevista(i)
        numAcertos = numAcertos + 1;
    end
end


% Calcular a matriz de confusão
conf_matrix = confusionmat(Y_validacao, classePrevista);

% Plotar a matriz de confusão
figure;
bar3(conf_matrix);
title('Matriz de Confusão');
xlabel('Classe Prevista');
ylabel('Classe Verdadeira');
zlabel('Contagem');
