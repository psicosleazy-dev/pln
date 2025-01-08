# Experimentação e Análise Comparativa de Modelos para Tradução Automática de Reportagens Científicas

## Autores
- Lucas M. D. Brum¹
- Guilherme R. Rodrigues²
- Santiago Lühring³  

¹Instituto de Informática – Universidade Federal do Rio Grande do Sul (UFRGS)  
Caixa Postal 15.064 – 91.501-970 – Porto Alegre – RS – Brazil  
{lucas.brum, grrodrigues, slbluhring}@inf.ufrgs.br  

---

## Resumo  
Este repositório acompanha o artigo que detalha a experimentação e análise comparativa de desempenho entre os modelos generativos **T5**, **mBart** e **MarianMT** (em duas versões: **Romance** e **Big**) para a tradução automática de sentenças de textos científicos. O estudo utilizou reportagens da revista da Fundação de Amparo à Pesquisa do Estado de São Paulo (**FAPESP**) traduzidas do inglês para o português.  

As traduções geradas foram avaliadas com as métricas **BLEU**, **METEOR**, **ROUGE** e uma análise qualitativa baseada em avaliação humana. O trabalho também mensurou o tempo de processamento de cada modelo.  

---

## Introdução  
A **tradução automática (TA)** é uma ferramenta poderosa para transpor barreiras linguísticas, permitindo a conversão de conteúdos entre idiomas distintos. Este projeto investiga a eficiência de quatro modelos de tradução automática na tarefa de traduzir textos científicos:  
- **T5**  
- **mBart**  
- **MarianMT Romance**  
- **MarianMT Big**  

A análise considera não apenas métricas quantitativas (BLEU, METEOR, ROUGE), mas também a percepção humana sobre a qualidade semântica e precisão das traduções.  

Os resultados indicaram que, enquanto os modelos **T5** e **MarianMT Big** produziram traduções mais consistentes, o **mBart** apresentou dificuldades, como métricas mais baixas (especialmente BLEU) e maior tempo de processamento.  

---

## Principais Resultados  
- **Desempenho por métrica:**  
  - BLEU foi usado para verificar precisão por correspondência de n-gramas.  
  - ROUGE avaliou a cobertura semântica do conteúdo traduzido.  
  - METEOR forneceu um equilíbrio entre precisão e recall.  
- **Tempo de Execução:**  
  - Tradução de 143 sentenças levou cerca de **50 minutos**.  
  - Tradução de 107 sentenças levou cerca de **38 minutos**.  
- **Comparação entre Modelos:**  
  - O **MarianMT Big** apresentou melhor desempenho geral.  
  - O **mBart** teve a maior demora no processamento e as menores pontuações em BLEU.
---

## Estrutura do Repositório  
- **/data**: Conjunto de dados utilizado nas experimentações.  
- **/results**: Resultados das traduções, incluindo métricas e análise qualitativa.  
- **/human_eval**: Scripts e resultados da avaliação humana automatizada.

---

## Como Reproduzir os Experimentos  
1. Clone o repositório:  
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio