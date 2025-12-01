"""
Módulo de Machine Learning para Previsão de Escanteios.

Este módulo é o "Estudante" do sistema. Ele usa algoritmos de Inteligência Artificial
para aprender com o passado e prever o futuro.

Conceitos Principais:
---------------------
1. **Random Forest (Floresta Aleatória)**:
   Imagine que você pergunta para 100 especialistas se vai chover.
   Se 80 disserem "Sim", você confia. O Random Forest faz isso:
   cria 100 "árvores de decisão" e faz uma votação.

2. **Treinamento**:
   O processo de mostrar milhares de jogos passados para o computador
   aprender as correlações (ex: "Time que chuta muito = Mais escanteios").

3. **Métricas (MAE e R²)**:
   - MAE: "Em média, quantos escanteios o modelo erra?"
   - R²: "O quanto o modelo entende do jogo?" (0 a 1)

Regras de Negócio:
------------------
- O modelo é salvo em um arquivo (.pkl) para não precisar treinar toda vez.
- Usamos 80% dos dados para treinar e 20% para testar (prova final).
"""

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class CornerPredictor:
    """
    Modelo de Machine Learning para previsão de escanteios em partidas de futebol.

    Utiliza Random Forest Regressor para prever o total de escanteios de uma
    partida com base nas médias históricas dos times.

    Regras de Negócio:
        - Random Forest com 100 árvores (n_estimators=100)
        - Seed fixa (random_state=42) para reprodutibilidade
        - Divisão treino/teste: 80%/20%
        - Modelo salvo em arquivo .pkl para reuso

    Algoritmo Random Forest:
        O Random Forest é um ensemble de árvores de decisão que:
        1. Cria múltiplas árvores com subamostras aleatórias dos dados
        2. Cada árvore vota na previsão final
        3. A média das previsões é o resultado (regressão)

        Vantagens para escanteios:
        - Lida bem com features não-lineares
        - Robusto a outliers (jogos atípicos)
        - Captura interações entre features (ex: time ofensivo vs defensivo)

    Attributes:
        model_path (str): Caminho para salvar/carregar o modelo.
        model: Instância do RandomForestRegressor.

    Example:
        >>> predictor = CornerPredictor()
        >>> predictor.train(X_train, y_train)
        >>> prediction = predictor.predict(X_new)
    """

    def __init__(self, model_path: str = "data/corner_model.pkl"):
        """
        Inicializa o preditor com modelo padrão.

        Args:
            model_path: Caminho para salvar/carregar modelo treinado.

        Lógica:
            1. Define caminho de persistência
            2. Inicializa Random Forest com 100 árvores
            3. Define seed para reprodutibilidade

        Parâmetros do Random Forest:
            - n_estimators=100: 100 árvores no ensemble
            - random_state=42: Seed fixa para resultados reproduzíveis
        """
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X, y) -> tuple:
        """
        Treina o modelo Random Forest com os dados históricos.

        Divide os dados em treino/teste, treina o modelo e avalia performance.

        Args:
            X: Features de entrada (DataFrame ou array 2D).
               Colunas esperadas:
               - home_avg_corners, home_avg_shots, home_avg_goals
               - away_avg_corners, away_avg_shots, away_avg_goals
            y: Target - total de escanteios da partida (Series ou array 1D).

        Returns:
            tuple: (MAE, R²) - Métricas de avaliação no conjunto de teste.

        Lógica:
            1. Divide dados em 80% treino, 20% teste
            2. Treina Random Forest no conjunto de treino
            3. Faz previsões no conjunto de teste
            4. Calcula MAE e R² para avaliação
            5. Salva modelo treinado em disco

        Métricas:
            - MAE (Mean Absolute Error): Erro médio absoluto.
              Interpretação: "Em média, erramos por X escanteios"
              Bom valor: MAE < 2.0 escanteios

            - R² (Coeficiente de Determinação): Variância explicada.
              Interpretação: "O modelo explica X% da variação"
              Bom valor: R² > 0.3 para dados de futebol

        Cálculos:
            ```
            MAE = (1/n) * Σ|y_real - y_previsto|

            R² = 1 - (SS_res / SS_tot)
            onde:
                SS_res = Σ(y_real - y_previsto)²
                SS_tot = Σ(y_real - média(y))²
            ```

        Example:
            >>> mae, r2 = predictor.train(X, y)
            >>> print(f"MAE: {mae:.2f}, R²: {r2:.2f}")
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Treinando modelo...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Modelo treinado! MAE: {mae:.2f}, R2: {r2:.2f}")

        self.save_model()
        return mae, r2

    def predict(self, X_new) -> list:
        """
        Faz previsão de escanteios para novas partidas.

        Utiliza o modelo treinado para prever o total de escanteios
        esperado com base nas features dos times.

        Args:
            X_new: Features da nova partida (array 2D).
                  Formato: [[home_avg_corners, home_avg_shots, home_avg_goals,
                            away_avg_corners, away_avg_shots, away_avg_goals]]

        Returns:
            list: Array com previsões (1 valor por partida).

        Lógica:
            1. Passa features pelo modelo treinado
            2. Cada árvore do Random Forest faz sua previsão
            3. Média das 100 árvores é o resultado final

        Regras de Negócio:
            - Modelo deve estar treinado ou carregado antes
            - Features devem estar na mesma ordem do treino
            - Resultado é valor contínuo (ex: 10.45 escanteios)

        Example:
            >>> X_new = [[5.5, 4.0, 1.5, 4.0, 3.5, 1.2]]
            >>> pred = predictor.predict(X_new)
            >>> print(f"Previsão: {pred[0]:.1f} escanteios")
        """
        return self.model.predict(X_new)

    def save_model(self) -> None:
        """
        Persiste o modelo treinado em disco.

        Utiliza joblib para serialização eficiente de objetos scikit-learn.

        Lógica:
            1. Serializa modelo usando joblib.dump()
            2. Salva em arquivo .pkl no caminho especificado

        Regras de Negócio:
            - Arquivo .pkl é binário e não editável manualmente
            - Modelo pode ser carregado posteriormente com load_model()
            - Diretório 'data/' deve existir
        """
        joblib.dump(self.model, self.model_path)
        print(f"Modelo salvo em {self.model_path}")

    def load_model(self) -> bool:
        """
        Carrega modelo previamente treinado do disco.

        Returns:
            bool: True se modelo carregado com sucesso, False se não encontrado.

        Lógica:
            1. Tenta carregar arquivo .pkl com joblib.load()
            2. Se sucesso, substitui modelo atual
            3. Se arquivo não existe, retorna False

        Regras de Negócio:
            - Modelo deve ter sido treinado e salvo anteriormente
            - Versão do scikit-learn deve ser compatível
            - Retorna False silenciosamente se arquivo não existe

        Example:
            >>> if predictor.load_model():
            ...     pred = predictor.predict(X_new)
            ... else:
            ...     print("Treine o modelo primeiro!")
        """
        try:
            self.model = joblib.load(self.model_path)
            print("Modelo carregado com sucesso.")
            return True
        except FileNotFoundError:
            print("Modelo não encontrado. É necessário treinar primeiro.")
            return False
