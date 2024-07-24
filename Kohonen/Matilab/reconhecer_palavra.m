% Carregar os dados DIREITA!!
load('dados_audio_DI_teste.mat');
load('dados_audio_REI_teste.mat');
load('dados_audio_TA_teste.mat');

% Carregar os dados ESQUERDA!!
load('dados_audio_ES_teste.mat');
load('dados_audio_QUER_teste.mat');
load('dados_audio_DA_teste.mat');

function palavra_reconhecida = reconhece_palavra(silaba_1, silaba_2, silaba_3)

    palavra_reconhecida = '';

    % Chamada da função Saida_rede para a primeira sílaba
    classe_prevista = Saida_rede(silaba_1);
    % Verifica a classe prevista e atualiza a palavra reconhecida
    if classe_prevista == 1
        disp('Sílaba DI reconhecida');
        classe_prevista = Saida_rede(silaba_2);
        if classe_prevista == 2
            disp('Sílaba REI reconhecida');
            % Chamada da função Saida_rede para a terceira sílaba
            classe_prevista = Saida_rede(silaba_3);
            if classe_prevista == 3
                disp('Sílaba TA reconhecida');
                palavra_reconhecida = 'DIREITA';
            end    
        end
    elseif classe_prevista == 4
        disp('Sílaba ES reconhecida');
        classe_prevista = Saida_rede(silaba_2);
        if classe_prevista == 5
            disp('Sílaba QUER reconhecida');
            % Chamada da função Saida_rede para a terceira sílaba
            classe_prevista = Saida_rede(silaba_3);
            if classe_prevista == 6
                disp('Sílaba DA reconhecida');
                palavra_reconhecida = 'ESQUERDA';
            end    
        end
    else
        palavra_reconhecida = 'PALAVRA NÃO RECONHECIDA';
    end
end

function classePrevista = Saida_rede(X)
    load('pesos_bias_treinados.mat');

    numAmostras = size(X, 1);
    numNeuronios = size(pesos, 2);
    vencedores = zeros(numAmostras, 1);

    % Encontrar os neurônios vencedores para cada amostra de entrada
    for i = 1:numAmostras
        distancias = sum((pesos - X(i, :)').^2, 1);
        [~, vencedor] = min(distancias);
        vencedores(i) = vencedor;
    end

    % Prever a classe para cada amostra de entrada
    classePrevista = zeros(numAmostras, 1);
    for i = 1:numAmostras
        neuronio = vencedores(i);
        classePrevista(i) = rotuloMaisComum(neuronio);
    end
end


% Conjunto de amostras de silabas X
X_silaba_1 = cat(1, dados_audio_di_teste{:});
X_silaba_2 = cat(1, dados_audio_rei_teste{:});
X_silaba_3 = cat(1, dados_audio_ta_teste{:});

% Atribuindo as três sílabas 
silaba_1 = X_silaba_1(3,:); % usando a amostra Nº3
silaba_2 = X_silaba_2(5,:); % usando a amostra Nº5 
silaba_3 = X_silaba_3(2,:); % usando a amostra Nº1


X_silaba_4 = cat(1, dados_audio_es_teste{:});
X_silaba_5 = cat(1, dados_audio_quer_teste{:});
X_silaba_6 = cat(1, dados_audio_da_teste{:});

% Atribuindo as três sílabas 
silaba_4 = X_silaba_4(1,:); % usando a amostra Nº1
silaba_5 = X_silaba_5(4,:); % usando a amostra Nº5 
silaba_6 = X_silaba_6(2,:); % usando a amostra Nº2

fprintf('Treinamento - Kohonen - reconhecimento palavras\n');
% Chamada da função reconhece_palavra com as três sílabas teste 01
palavra_reconhecida = reconhece_palavra(silaba_1, silaba_2 , silaba_3);

% Exibe a palavra reconhecida
disp(['Saída: ', palavra_reconhecida]);

fprintf('__________________Teste 02____________________________\n');

% Chamada da função reconhece_palavra com as três sílabas teste 02
palavra_reconhecida = reconhece_palavra(silaba_4, silaba_5 , silaba_6);

% Exibe a palavra reconhecida
disp(['Saída: ', palavra_reconhecida]);