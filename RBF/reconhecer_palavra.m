
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

function classe_prevista = Saida_rede(X_teste)
    % Carregar os pesos e bias treinados
    load('pesos_bias_treinados_RBF_WTA.mat');
    k = 1;
    % Normalização para [0,1] 
    X_teste = (X_teste - min(X_teste(:))) / (max(X_teste(:)) - min(X_teste(:)));
    
    % Calculando a saída da camada RBF
    Yh_test = zeros(size(X_teste, 1), numRBFNeurons);
    for i = 1:numRBFNeurons
        mu_i = sqrt(sum((X_teste - centers(i, :)).^2, 2)); % Calculo de mu_i
        Yh_test(:, i) = exp(-mu_i.^2 / (2 * sigma(i)^2)); % Função Gaussiana
    end
    
    % 1 - Calcular a entrada da camada de saída
    net_o_test = Woh * Yh_test' + bias_oh * ones(1, size(Yh_test', 2));
    
    % 2 - Calcular a saída da rede neural
    Ys_test = k * net_o_test';
    
    % 3 - Classificar as saídas (classe com a maior probabilidade)
    [~, classe_prevista] = max(Ys_test, [], 2);
end

% Selecionar aleatoriamente as  sílabas e testa se as palavras foram
% reconhecidas.
function selecionar_e_testar_silabas(num_direita, num_esquerda)
    % Carregar os dados DIREITA e ESQUERDA!!
    load('dados_audio_DI_teste.mat');
    load('dados_audio_REI_teste.mat');
    load('dados_audio_TA_teste.mat');
    load('dados_audio_ES_teste.mat');
    load('dados_audio_QUER_teste.mat');
    load('dados_audio_DA_teste.mat');

    % Conjunto de amostras de sílabas ***DIREITA***
    X_silaba_1 = cat(1, dados_audio_di_teste{:});
    X_silaba_2 = cat(1, dados_audio_rei_teste{:});
    X_silaba_3 = cat(1, dados_audio_ta_teste{:});

    % Conjunto de amostras de sílabas ***ESQUERDA***
    X_silaba_4 = cat(1, dados_audio_es_teste{:});
    X_silaba_5 = cat(1, dados_audio_quer_teste{:});
    X_silaba_6 = cat(1, dados_audio_da_teste{:});

    % Selecionando as amostras aleatórias, temos.
    for i = 1:num_direita
        idx1 = randi(size(X_silaba_1, 1));
        idx2 = randi(size(X_silaba_2, 1));
        idx3 = randi(size(X_silaba_3, 1));

        silaba_1 = X_silaba_1(idx1, :);
        silaba_2 = X_silaba_2(idx2, :);
        silaba_3 = X_silaba_3(idx3, :);

        % Teste de reconhecimento para DIREITA
        palavra_reconhecida = reconhece_palavra(silaba_1, silaba_2, silaba_3);
        disp(['Teste DIREITA ', num2str(i), ' - Saída: ', palavra_reconhecida]);
            fprintf('__________________\n');
    end

    fprintf('****************************\n');

    % Selecionar amostras aleatórias para ESQUERDA
    for j = 1:num_esquerda
        idx4 = randi(size(X_silaba_4, 1));
        idx5 = randi(size(X_silaba_5, 1));
        idx6 = randi(size(X_silaba_6, 1));

        silaba_4 = X_silaba_4(idx4, :);
        silaba_5 = X_silaba_5(idx5, :);
        silaba_6 = X_silaba_6(idx6, :);

        % Teste de reconhecimento para ESQUERDA
        palavra_reconhecida = reconhece_palavra(silaba_4, silaba_5, silaba_6);
        disp(['Teste ESQUERDA ', num2str(j), ' - Saída: ', palavra_reconhecida]);
        fprintf('__________________\n');
    end
end

% Testando
num_direita = 5;  %numero de teste direita
num_esquerda = 5; %numero de teste esquerda
selecionar_e_testar_silabas(num_direita, num_esquerda);