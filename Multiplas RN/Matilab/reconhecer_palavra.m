
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

function classe_prevista = Saida_rede(X)    
    load('pesos_bias_treinados.mat');
    Ys_teste_final = 0;

    % Testar com cada rede treinada
    for rede = 1:NRedes
        % Obter pesos e biases da rede atual
        Whi = Whi_all{rede};
        Woh = Woh_all{rede};
        bias_hi = bias_hi_all{rede};
        bias_oh = bias_oh_all{rede};

        % Passo 2: Calcular entrada da camada escondida
        net_h_teste = Whi * X' + bias_hi * ones(1, size(X', 2));

        % Passo 3: Calcular a saída da camada escondida
        Yh_teste = logsig(net_h_teste);

        % Passo 4: Calcular entrada da camada de saída
        net_o_teste = Woh * Yh_teste + bias_oh * ones(1, size(Yh_teste, 2));

        % Passo 5: Calcular a saída da rede neural
        Ys_teste = logsig(net_o_teste);

        % Atualizar a saída final acumulada
        Ys_teste_final = Ys_teste_final + Ys_teste;
    end

    % Calcular a classe prevista para cada amostra
    [~, classe_prevista] = max(Ys_teste_final, [], 1);
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