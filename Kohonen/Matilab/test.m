% Carregar os dados DIREITA!!
load('dados_audio_DI_teste.mat');
load('dados_audio_REI_teste.mat');
load('dados_audio_TA_teste.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES_teste.mat');
load('dados_audio_QUER_teste.mat');
load('dados_audio_DA_teste.mat');

load('pesos_bias_treinados.mat');

dados_teste_di = cat(1, dados_audio_di_teste{:});
dados_teste_rei= cat(1, dados_audio_rei_teste{:});
dados_teste_ta = cat(1, dados_audio_ta_teste{:});

dados_teste_es = cat(1, dados_audio_es_teste{:});
dados_teste_quer = cat(1, dados_audio_quer_teste{:});
dados_teste_da = cat(1, dados_audio_da_teste{:});

%Temos 6 CLASSES; DI, REI, TA, ES, QUER, DA
X_teste = [dados_teste_di; dados_teste_rei; dados_teste_ta; dados_teste_es;  dados_teste_quer; dados_teste_da];

Y_teste = [1*ones(size(dados_teste_di', 2), 1); 2*ones(size(dados_teste_rei', 2), 1); 3*ones(size(dados_teste_ta', 2), 1);
        4*ones(size(dados_teste_es', 2), 1); 5*ones(size(dados_teste_quer', 2), 1); 6*ones(size(dados_teste_da', 2), 1)];


% Testar a rede
numAmostrasTeste = size(X_teste, 1);
numNeuronios = size(pesos, 2);
vencedores = zeros(numAmostrasTeste, 1);

% Encontrar os neurônios vencedores para cada amostra de teste.
for i = 1:numAmostrasTeste
    distancias = sum((pesos - X_teste(i, :)').^2, 1);
    [~, vencedor] = min(distancias);
    vencedores(i) = vencedor;
end

rotulosTeste = Y_teste;
numClasses = length(unique(rotulosTeste));
rotuloMaisComum = zeros(numNeuronios, 1);

% Encontrar o rótulo mais comum para cada neurônio, temos:
for neuronio = 1:numNeuronios
    indicesNeuronio = find(vencedores == neuronio);
    classesNeuronio = rotulosTeste(indicesNeuronio);
    if ~isempty(classesNeuronio)
        rotuloMaisComum(neuronio) = mode(classesNeuronio);
    end
end

numAcertos = 0;
classePrevista = zeros(numAmostrasTeste, 1);

% Prever a classe para cada amostra de teste
for i = 1:numAmostrasTeste
    neuronio = vencedores(i);
    classePrevista(i) = rotuloMaisComum(neuronio);
    if rotulosTeste(i) == classePrevista(i)
        numAcertos = numAcertos + 1;
    end
end

taxaAcerto = numAcertos / numAmostrasTeste;

% Calcular a matriz de confusão e métricas de desempenho
confMat = confusionmat(rotulosTeste, classePrevista);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Médias
precisaoMedia = mean(precision, 'omitnan');
recallMedio = mean(recall, 'omitnan');
f1scoreMedio = mean(f1score, 'omitnan');

fprintf('\nNúmero de amostras: %d\n', numAmostrasTeste);
fprintf('Número de acertos: %d\n', numAcertos);
fprintf('Taxa de acerto: %.2f%%\n', taxaAcerto * 100);
fprintf('Precisão média: %.2f\n', precisaoMedia);
fprintf('Recall médio: %.2f\n', recallMedio);
fprintf('F1-Score médio: %.2f\n', f1scoreMedio);

disp('Matriz de confusão:')
disp(confMat);

save('pesos_bias_treinados.mat', 'pesos', 'rotuloMaisComum');