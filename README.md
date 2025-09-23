# 🪼 Synapse: BioAsk RAG Client

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)

<div align="center">
  <img src="docs/logo.png" alt="Synapse Logo" width="400"/>
</div>

---
**Synapse: BioAsk RAG Client — Seu assistente de IA especializado em Medicina Baseada em Evidências**

O Synapse é um sistema de Geração Aumentada por Recuperação (RAG) altamente sofisticado, projetado para responder a perguntas complexas em saúde e biomedicina com base em artigos médicos revisados por pares.

O objetivo é oferecer respostas contextualizadas, concisas e fundamentadas em evidências científicas, sempre com referências diretas às fontes consultadas. Dessa forma, estudantes de medicina, médicos, pesquisadores e pacientes têm acesso rápido e confiável a informações validadas, promovendo uma prática médica mais segura e informada.

O sistema foi treinado com datasets [BioAsk](https://www.bioasq.org/), abrangendo publicações dos últimos cinco anos (2020–2025).

---
## 🚀 Arquitetura e Tecnologias

O coração do Guardião é um grafo computacional gerenciado pelo **LangGraph**, que orquestra a colaboração entre diferentes agentes especializados.

**Fluxo de Agentes Detalhado:**

1.  **Agente Supervisor**: Atua como o roteador principal. Ele analisa a pergunta e a classifica em uma das categorias: `pergunta_sobre_previdencia`, `saudacao`, `meta_pergunta` ou `fora_de_topico`.
2.  **Execução Condicional**:
    -   Se for uma **saudação** ou **meta-pergunta**, agentes específicos fornecem respostas diretas e amigáveis.
    -   Se for **fora de tópico**, um agente recusa educadamente a solicitação.
    -   Se for sobre **Direito Previdenciário**, a pergunta entra no pipeline RAG principal.
3.  **Pipeline RAG**:
    -   **Query Transformer**: Expande a pergunta original em múltiplas variantes para uma busca mais abrangente.
    -   **Retriever & Reranker**: Busca os documentos mais relevantes usando FAISS e os reordena com um Cross-Encoder para máxima precisão.
    -   **Answerer**: O LLM (Gemini) gera uma resposta com base nos documentos reordenados, incluindo citações.
    -   **Self-Check & Safety**: Agentes finais verificam a presença de fontes e adicionam um aviso legal antes de entregar a resposta final.

**Pipeline de Recuperação Avançada:**
`Consulta` → `Busca Vetorial (FAISS)` → `Top 20 Candidatos` → `Cross-Encoder Reranker` → `Top 5 Mais Relevantes` → `LLM (Gemini)`

| Componente              | Tecnologia/Modelo Utilizado                            | Propósito                                                                      |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Orquestração** | `LangChain` + `LangGraph`                              | Gerencia o fluxo de trabalho condicional entre os agentes.                     |
| **LLM** | `Google Gemini (gemini-2.5-flash)`                     | Geração de texto, reescrita de perguntas e classificação de intenção.          |
| **Embeddings** | `thenlper/gte-small`                                   | Cria representações vetoriais dos trechos de texto para busca semântica.       |
| **Vector Store** | `FAISS (Facebook AI Similarity Search)`                | Armazena os vetores e realiza a busca por similaridade de forma eficiente.     |
| **Reranker** | `BAAI/bge-reranker-base` (Cross-Encoder)               | Reordena os resultados da busca inicial para máxima precisão.                  |
| **Interface (UI)** | `Streamlit`                                            | Cria a interface web interativa para o usuário.                                |
| **Containerização** | `Docker`                                               | Garante a reprodutibilidade e facilita o deploy do ambiente.                   |

---
## ⚙️ Como Executar o Projeto

Siga os passos abaixo para configurar e rodar o Guardião dos Direitos em seu ambiente.

### **Pré-requisitos**

-   Python 3.10
-   Docker e Docker Compose (para a execução com container)
-   Uma chave de API do Google AI Studio (para o Gemini)

### **1. Configuração Inicial**

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/mttgermano/synapse.git
    cd synapse
    ```

2.  **Crie o arquivo de ambiente:**
    Copie o arquivo `.env.example` para `.env` e adicione sua chave de API do Google.
    ```bash
    cp .env.example .env
    # Edite o arquivo .env e insira sua chave
    # API_KEY="sua_chave_aqui"
    ```

### **2. Execução (Duas Opções)**

#### **Opção A: Ambiente Local (Recomendado para Desenvolvimento)**

1.  **Crie o ambiente virtual e instale as dependências:**
    O comando `setup` fará isso por você.
    ```bash
    make setup
    ```

2.  **Ative o ambiente virtual:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Inicie a Aplicação:**
    ```bash
    make run
    ```
    Acesse a interface em `http://localhost:8501`.

#### **Opção B: Usando Docker (Recomendado para Produção)**

1.  **Construa a Imagem Docker:**
    ```bash
    make build
    ```

2.  **Execute o Container:**
    Este comando irá iniciar o container, mapeando a porta e o diretório do projeto.
    ```bash
    make run-docker
    ```
    Acesse a interface em `http://localhost:8501`.

---

## 📜 Comandos do Makefile

Um `Makefile` foi configurado para simplificar as tarefas comuns.

| Comando         | Descrição                                                                           |
| --------------- | ----------------------------------------------------------------------------------- |
| `make setup`    | Cria um ambiente virtual `.venv` e instala todas as dependências do `requirements.txt`. |
| `make run`      | Inicia a aplicação Streamlit no ambiente local.                                     |
| `make build`    | Constrói a imagem Docker para a aplicação.                                          |
| `make run-docker` | Executa a aplicação dentro de um container Docker.                                |
| `make evaluate` | Avalia o sistema RAG usando o CSV de teste e gera relatórios em JSON e Markdown.    |
<!--
---

## 📊 Avaliação de Desempenho

Para garantir a qualidade e a confiabilidade das respostas, o **Guardião dos Direitos** foi submetido a uma avaliação rigorosa utilizando o framework **Ragas**. O benchmark foi realizado com um conjunto de **10 perguntas** complexas sobre regras de aposentadoria e benefícios, avaliando quatro métricas essenciais.

| Métrica            | Média   | Status | Análise                                                                                 |
| :----------------- | :-----: | :----: | :-------------------------------------------------------------------------------------- |
| **Answer Relevancy** | `0.972` | 🟢   | As respostas estão **excelentemente alinhadas** com as perguntas dos usuários.          |
| **Context Precision**| `0.876` | 🟢   | O sistema é **muito eficiente** em recuperar os contextos mais relevantes.             |
| **Context Recall**   | `0.833` | 🟢   | O sistema consegue encontrar a **maioria dos contextos necessários** para uma resposta completa.|
| **Faithfulness**     | `0.702` | 🟡   | As respostas melhoraram, mas o foco continua em **garantir a total aderência** aos fatos. |

### Análise dos Resultados

Os resultados demonstram uma **evolução positiva** no desempenho geral do sistema, com uma performance quase perfeita em **Answer Relevancy** (`0.972`) e uma melhoria notável em **Faithfulness** (`0.702`). O sistema de recuperação continua robusto, mantendo altas pontuações de precisão e recall.

### Resultados detalhados

Os resultados completos, incluindo métricas agregadas, respostas individuais e os contextos utilizados, estão disponíveis no relatório:  
[📑 eval/ragas_report.md](eval/ragas_report.md)

---
-->
## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 📚 Citação

Se você utilizar o Synapse em pesquisas ou trabalhos, cite este projeto.  
[`CITATION.cff`](./docs/CITATION.cff)
```bibtex
@software{synapse2025,
  author = {Matheus Germano, Pedro Simões},
  title = {Synapse: BioAsk RAG Client},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {[https://github.com/mttgermano/synapse](https://github.com/mttgermano/synapse)}
}
```
---
