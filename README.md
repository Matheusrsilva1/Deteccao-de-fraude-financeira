# Deteccao-de-fraude-financeira

Objetivo Principal:
Desenvolver um sistema de Machine Learning (ML) completo, funcional e avançado para identificar transações financeiras potencialmente fraudulentas. O sistema deve ser treinado usando dados sintéticos realistas (que você criará com cenários específicos de fraude), incorporar engenharia de features avançada, utilizar modelos de classificação potentes (como XGBoost/LightGBM), otimizar hiperparâmetros, avaliar o desempenho com foco em métricas de negócio (incluindo otimização de limiar de decisão), e fornecer interpretabilidade para as previsões usando SHAP. O projeto deve ser construído do zero até um estado funcional simulando predições em novas transações com explicações.

Descrição Detalhada do Projeto:
O objetivo é construir um pipeline de ML robusto que classifique transações financeiras como "Legítima" (0) ou "Fraudulenta" (1). Como não há dados reais, o primeiro passo é gerar um dataset sintético de alta qualidade, incluindo múltiplos cenários de fraude e algum ruído. O projeto deve então cobrir engenharia de features (temporal e comportamental), pré-processamento cuidadoso, tratamento de desbalanceamento de classes, treinamento e otimização de um modelo de gradient boosting, avaliação detalhada com otimização de limiar, e a criação de uma função de predição que não só classifique novas transações, mas também explique por que uma decisão foi tomada usando SHAP.

Requisitos Técnicos:

Linguagem: Python 3.x

Bibliotecas Principais:

Pandas: Manipulação e geração de dados.

NumPy: Operações numéricas.

Scikit-learn: Pré-processamento, divisão de dados, RandomizedSearchCV (ou GridSearchCV), métricas de avaliação (classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc).

imbalanced-learn: Para tratamento de desbalanceamento (ex: SMOTE).

XGBoost OU LightGBM: Modelos de Gradient Boosting. Escolha um e use-o como modelo principal. (Incluir no requirements.txt).

SHAP: Para interpretabilidade do modelo (Explainable AI). (Incluir no requirements.txt).

Joblib: Para salvar/carregar o modelo e o pré-processador.

(Opcional) Matplotlib/Seaborn: Para visualizações (ex: curva Precision-Recall).

Passos Detalhados (Construir do Zero):

Passo 1: Geração Aprimorada de Dados Sintéticos (generate_enhanced_data.py)

Crie um script Python para gerar um dataset aprimorado de transações.

Formato: Salve como synthetic_transactions_enhanced.csv.

Tamanho: Gere um volume razoável (ex: 50.000 a 100.000 registros).

Colunas: Inclua transaction_id, user_id, timestamp, transaction_amount, transaction_type, merchant_category, device_used, location_country, e a variável alvo is_fraud (0 ou 1).

Lógica de Fraude Aprimorada:

Desbalanceamento: Mantenha a fraude rara (ex: 1-5% do total).

Implemente Múltiplos Cenários de Fraude Explícitos (Misture-os):

Valor Alto Atípico: Fraudes com transaction_amount > X vezes a média geral/do usuário implícito.

Velocidade: Sequências curtas de transações de baixo valor para o mesmo user_id/merchant_category, marcando algumas como fraude.

Localização Incomum: Defina um 'país base' implícito para usuários e gere fraudes com location_country diferente.

Horário Incomum: Aumente a chance de fraude para transações em horários atípicos (ex: madrugada).

Introduzir Ruído/Sutileza: Adicione algumas transações legítimas que pareçam suspeitas e algumas fraudes que não se encaixem perfeitamente nos cenários acima.

Passo 2: Engenharia de Features Avançada e Pré-processamento (preprocess_and_feature_engineer.py ou integrado no treino)

Carregue synthetic_transactions_enhanced.csv.

Engenharia de Features: Crie novas features antes de dividir os dados:

Temporais: hour_of_day, day_of_week.

Relativas ao Tempo: time_since_last_transaction_user (tempo desde a última tx do usuário, requer ordenação e groupby).

Frequência: transaction_frequency_last_24h_user (contagem de tx do usuário nas últimas 24h).

Comportamentais (Agregadas - CUIDADO COM DATA LEAKAGE):

user_avg_amount_history: Valor médio histórico das transações do user_id (calcule usando uma janela de tempo passada ou todo o histórico antes da tx atual se possível; se simplificar, calcule sobre todo o dataset antes do split, ciente da limitação).

amount_deviation_from_avg: transaction_amount / (user_avg_amount_history + 1e-6).

Pré-processamento:

Trate valores ausentes (que podem surgir da engenharia de features, ex: primeira transação do usuário). Preencha com 0, -1 ou uma mediana/média apropriada.

Codifique variáveis categóricas: Use OneHotEncoder (lidando com categorias desconhecidas na predição, handle_unknown='ignore').

Escale features numéricas: Use StandardScaler.

Pipeline: É altamente recomendado usar ColumnTransformer e Pipeline do Scikit-learn para organizar essas etapas e evitar erros.

Salve o Pré-processador: Salve o objeto Pipeline ou ColumnTransformer ajustado como preprocessor_enhanced.joblib.

Passo 3: Divisão de Dados e Tratamento de Desbalanceamento (train_evaluate_model.py)

Separe Features (X) e Alvo (y).

Divisão Treino/Teste: Divida X e y em conjuntos de treino (ex: 80%) e teste (ex: 20%), usando train_test_split com stratify=y.

Tratamento de Desbalanceamento (APENAS no Conjunto de Treino):

Aplique SMOTE do imbalanced-learn no X_train, y_train para gerar X_resampled, y_resampled. Use estes dados rebalanceados para treinar o modelo.

Passo 4: Treinamento, Comparação e Otimização de Modelo (train_evaluate_model.py)

Seleção de Modelo: Escolha XGBoost (XGBClassifier) ou LightGBM (LGBMClassifier) como o modelo principal. Opcionalmente, treine um LogisticRegression ou RandomForestClassifier como baseline para comparação inicial.

Otimização de Hiperparâmetros:

Use RandomizedSearchCV (preferível para velocidade) ou GridSearchCV com validação cruzada (ex: cv=3 ou cv=5).

Defina um espaço de busca de hiperparâmetros relevante para o modelo escolhido (ex: n_estimators, learning_rate, max_depth, subsample, colsample_bytree, etc.).

Importante: Ajuste (fit) o objeto RandomizedSearchCV nos dados de treino rebalanceados (X_resampled, y_resampled).

Defina a métrica de pontuação (scoring) para a otimização, por exemplo, 'f1' (para a classe positiva=1) ou 'roc_auc' ou 'recall'. Use F1 para este exercício.

Modelo Final: Obtenha o melhor estimador encontrado pela busca (search.best_estimator_).

Salve o Modelo Final: Salve este modelo otimizado como fraud_detection_model_enhanced.joblib.

Passo 5: Avaliação Avançada e Otimização de Limiar (train_evaluate_model.py)

Predições no Conjunto de Teste: Use o modelo final (best_estimator_) para prever probabilidades (predict_proba) no conjunto de teste original (X_test). Pegue a probabilidade da classe positiva (fraude).

Métricas Padrão (com limiar 0.5): Calcule e imprima a Matriz de Confusão, classification_report, e roc_auc_score usando o limiar padrão de 0.5 para ter uma linha de base.

Curva Precision-Recall: Calcule e imprima o auc(precision, recall) (AUC-PR). Opcionalmente, plote a curva PR.

Otimização do Limiar:

Itere sobre as probabilidades previstas no teste para diferentes limiares (ex: de 0.05 a 0.95).

Para cada limiar, classifique as transações e calcule o f1_score para a classe fraude (pos_label=1).

Encontre o optimal_threshold que maximiza este f1_score.

Avaliação Final com Limiar Ótimo:

Imprima o optimal_threshold encontrado.

Calcule e imprima novamente o classification_report e a Matriz de Confusão usando as classificações baseadas no optimal_threshold. Destaque a melhoria (ou mudança) no Recall e Precision da classe fraude.

Passo 6: Função de Predição "Real-Time" com Explicação SHAP (predict_enhanced.py)

Crie um script para simular a predição em uma nova transação.

Carregue: Carregue o modelo otimizado (fraud_detection_model_enhanced.joblib) e o pré-processador (preprocessor_enhanced.joblib).

Carregue o Limiar Ótimo: Carregue ou defina o optimal_threshold encontrado no Passo 5.

Função predict_fraud_enhanced(transaction_data):

Recebe dados de uma única transação (dicionário Python).

Transforma o dicionário em DataFrame.

Aplica exatamente o mesmo pré-processamento usando o preprocessor_enhanced carregado. (Atenção: Se features agregadas dependem de histórico, esta parte pode precisar de acesso a dados históricos ou usar valores padrão/médios como fallback na simulação).

Prevê a probabilidade de fraude (model.predict_proba(X_processed)[:, 1]).

Classifica como fraude (prediction = 1 if probability >= optimal_threshold else 0).

Integração SHAP:

Crie um shap.Explainer (ex: shap.TreeExplainer(model)).

Calcule os valores SHAP para a instância pré-processada: shap_values = explainer.shap_values(X_processed). (Para classificadores binários em TreeExplainer, pode retornar uma lista; pegue o índice da classe positiva, geralmente 1).

Crie um DataFrame com os nomes das features e seus valores SHAP correspondentes para a instância. Ordene por valor absoluto SHAP decrescente.

Extraia as top N (ex: 5) features e seus valores SHAP como explicação.

Retorne: Um dicionário contendo: is_fraud (0 ou 1), fraud_probability (float), threshold_used (float), risk_level (string opcional: 'LOW'/'MEDIUM'/'HIGH' baseado na probabilidade), e explanation (dicionário ou lista das top N features e seus valores SHAP).

Exemplo de Uso: Demonstre no script como chamar a função com uma transação de exemplo (dicionário) e imprima o resultado retornado de forma clara.

Passo 7: Estrutura do Projeto e Documentação Final

Organização: Organize o código em arquivos lógicos (generate_enhanced_data.py, train_evaluate_model.py, predict_enhanced.py, talvez um config.py para constantes).

Comentários: Adicione comentários claros explicando o código.

requirements.txt: Crie um arquivo listando todas as dependências (pandas, numpy, scikit-learn, imbalanced-learn, xgboost ou lightgbm, shap, joblib).

README.md: Crie um arquivo README.md detalhado:

Objetivo do projeto aprimorado.

Descrição dos dados sintéticos e cenários de fraude.

Engenharia de features implementada.

Modelo usado e processo de otimização.

Como configurar o ambiente (pip install -r requirements.txt).

Como executar os scripts na ordem correta.

Interpretação dos resultados da avaliação (incluindo limiar ótimo).

Como usar predict_enhanced.py e interpretar a saída, incluindo as explicações SHAP.

Entregáveis Esperados:

Código Python completo e funcional para todas as etapas.

Arquivo CSV: synthetic_transactions_enhanced.csv.

Arquivos salvos: preprocessor_enhanced.joblib, fraud_detection_model_enhanced.joblib.

Arquivo requirements.txt.

Considerações Finais Cruciais:

Data Leakage: Seja extremamente cuidadoso ao criar features agregadas baseadas no tempo para não vazar informação do futuro para o passado.

Imbalance Handling: Aplique SMOTE (ou similar) somente nos dados de treino, após a divisão treino/teste.

Optimized Threshold: Lembre-se que o limiar de 0.5 raramente é o ideal para problemas de fraude. Use o limiar otimizado para a classificação final.

SHAP: As explicações SHAP são vitais para a confiança e utilidade do modelo em um cenário real. Certifique-se de que a saída seja clara.