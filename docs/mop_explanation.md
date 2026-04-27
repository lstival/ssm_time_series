# Mixture of Prompts (MoP) — Explicação

## Intuição básica

O encoder SimCLR foi treinado para **entender** séries temporais — ele transforma qualquer série em um vetor de características (embedding). Mas entender não é o mesmo que prever. O problema é: como usar esse encoder **congelado** para prever o futuro em datasets nunca vistos?

A solução do MoP: aprender um **banco de "estilos de previsão"** (prompts), e para cada série, escolher automaticamente qual estilo usar — sem fine-tuning por dataset.

### Analogia

Pense num sommelier (encoder) que analisa um vinho e descreve suas características. O MoP é como um **menu de maridamentos** (prompts): dependendo do perfil do vinho descrito pelo sommelier, o MoP escolhe a combinação certa de sugestões para fazer a previsão final.

---

## Arquitetura passo a passo

```
Série temporal x (B, C, L)
        │
        ▼
┌─────────────────────────────┐
│  StandardScaler (loader)    │  ← normalização por dataset (fitado no train split)
│  normalize=True             │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Encoder SimCLR (congelado) │  → embedding temporal ze (B*C, D)
│  Visual Encoder (congelado) │  → embedding visual  zv (B*C, D)
└─────────────────────────────┘
        │  concat
        ▼
      z = [ze, zv]  ← dim D*2 = 256
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  ModuleOfPrompts (MoP)  — ÚNICO módulo treinável    │
│                                                     │
│  Prompt Keys   K (8, 256)  ← "assinaturas"         │
│  Prompt Values V (8, 256)  ← "receitas"            │
│                                                     │
│  1. scores  = z @ K.T          →  (B*C, 8)         │
│  2. weights = softmax(scores)  →  (B*C, 8)         │
│  3. prompt  = weights @ V      →  (B*C, 256)       │
│  4. z_combined = [z, prompt]   →  (B*C, 512)       │
│  5. z_out = MLP(512→512)       →  (B*C, 512)       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Heads por horizonte        │  ← também treináveis
│  head[96]:  Linear(512→96)  │
│  head[192]: Linear(512→192) │
│  head[336]: Linear(512→336) │
│  head[720]: Linear(512→720) │
└─────────────────────────────┘
        │
        ▼
   Previsão ŷ (B*C, H, 1)
```

---

## Exemplo concreto com números

Suponha `B=2` amostras, `C=3` canais, `enc_dim=128`, `L=336`.

```python
x.shape = (2, 3, 336)   # 2 amostras, 3 canais, 336 timesteps

# Encoder processa canal por canal → B*C = 6 séries univariadas
x_reshaped = (6, 1, 336)

ze = encoder(x_reshaped)  # → (6, 128)
zv = visual(x_reshaped)   # → (6, 128)
z  = cat([ze, zv])         # → (6, 256)

# MoP: encontra qual "prompt" melhor descreve cada série
scores  = z @ K.T          # (6, 256) @ (256, 8) → (6, 8)
weights = softmax(scores)  # (6, 8)  ex: [0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0]
prompt  = weights @ V      # (6, 8) @ (8, 256) → (6, 256)  (mistura ponderada)

z_combined = cat([z, prompt])  # (6, 512)
z_out = proj(z_combined)       # (6, 512)

# Head para H=96:
pred = head[96](z_out)  # (6, 96) → reshape → (6, 96, 1)
```

Os pesos do softmax determinam **qual combinação de prompts** melhor representa aquela série. Uma série com forte sazonalidade diária pode ativar o "prompt de sazonalidade"; uma série com tendência linear ativa o "prompt de tendência".

---

## O que é treinado vs congelado

| Componente | Parâmetros | Status | Função |
|-----------|-----------|--------|--------|
| SimCLR encoder | ~1M | **Congelado** | Extrai features temporais |
| Visual encoder | ~1M | **Congelado** | Extrai features visuais |
| Prompt Keys K | 8×256 = 2K | **Treinável** | "Assinaturas" de padrões |
| Prompt Values V | 8×256 = 2K | **Treinável** | "Receitas" por padrão |
| Proj MLP | ~512K | **Treinável** | Combina z + prompt |
| Heads ×4 | ~4×512×H | **Treinável** | Previsão por horizonte |

O MoP tem **muito poucos parâmetros treináveis** (~600K) comparado ao encoder (2M+). Isso é a chave para zero-shot: aprende rápido e generaliza.

---

## Implementação no projeto

### Arquivos principais

| Arquivo | Papel |
|---------|-------|
| `src/models/mop_forecast.py` | `ModuleOfPrompts` + `MoPForecastModel` — toda a arquitetura |
| `src/experiments/mop_tuning_icml_loo.py` | Loop LOO: treina 1 MoP por fold, avalia no held-out |
| `src/experiments/mop_tuning_icml_only.py` | Treina 1 MoP em todos os 7 datasets (não zero-shot real) |
| `src/scripts/ssl/anunna_mop_zeroshot_icml_loo.sh` | SLURM job — 100 epochs, norm_mode=identity |

### Classes em `mop_forecast.py`

**`RevIN`** — Reversible Instance Normalization (normalização por instância, reversível).

**`ModuleOfPrompts`** — o núcleo do MoP: aprende K prompts e faz routing via atenção softmax.

**`MoPForecastModel`** — orquestra encoders congelados + MoP + heads de horizonte. Suporta `norm_mode` configurável:
- `'identity'` (atual no LOO): só o StandardScaler do loader normaliza — evita dupla normalização
- `'revin'`: aplica RevIN por instância em cima do loader — causa dupla normalização (evitar no LOO)
- `'minmax'`: normalização min-max por instância

---

## Variantes experimentadas

| Variante | Arquivo CSV | Descrição | Status |
|----------|-------------|-----------|--------|
| `mop_zeroshot_icml_only` | `mop_zeroshot_icml_only_results.csv` | Treina e avalia nos mesmos 7 datasets — **não é zero-shot real** | MSE ETTs ~0.148 |
| `mop_zeroshot_loo` | `mop_zeroshot_loo_results.csv` | LOO verdadeiro: nunca vê o dataset-alvo | MSE ETTs ~0.605 (em melhoria) |
| `mop_zeroshot_trainnorm` | `mop_zeroshot_trainnorm_results.csv` | Normalização manual no input — RevIN duplo quebrado | MSE ~200 (inválido) |
| `mop_zeroshot` (base) | `mop_zeroshot_results.csv` | Checkpoint antigo, scaler incompatível | MSE ~2.4 (inválido) |

---

## LOO: por que é zero-shot real

```
Fold 1: treina em [ETTm2, ETTh1, ETTh2, Weather, Traffic, Electricity]
         avalia em [ETTm1]  ← nunca visto durante treino do MoP

Fold 2: treina em [ETTm1, ETTh1, ETTh2, Weather, Traffic, Electricity]
         avalia em [ETTm2]  ← nunca visto durante treino do MoP
...
```

A hipótese: se os prompts aprenderam padrões **transferíveis** (tendência, sazonalidade, ruído), eles funcionam em domínios novos sem fine-tuning. O encoder permanece sempre congelado — só os prompts e heads são retreinados por fold.

---

## Problema de dupla normalização (bug corrigido em 2026-04-17)

Com `norm_mode='revin'` + loader `normalize=True`, havia dupla normalização:

```
x (bruto)
  → StandardScaler (loader, fitado no train split do dataset-alvo)
  → RevIN (MoP, per-instance, fitado no batch atual)
  → encoder
  → ... previsão em espaço duplamente normalizado
```

O MSE era calculado contra `y` que vinha do loader (espaço StandardScaler), mas as predições saíam do espaço RevIN-sobre-StandardScaler — inconsistência de escala.

**Fix**: `norm_mode='identity'` — o MoP não aplica normalização interna, confiando apenas no StandardScaler do loader, que é fitado corretamente no train split de cada dataset.

---

## Por que treinar o MoP com LOTSA + Chronos melhoraria o zero-shot

O encoder SimCLR foi treinado no corpus LOTSA + Chronos (~57 datasets). O espaço de embeddings que ele aprendeu "conhece" todos esses domínios. Os prompts atuais, porém, foram treinados apenas nos 7 datasets ICML — cobrindo uma fração pequena do espaço.

```
Espaço de embeddings (simplificado 2D):

  MoP atual (7 datasets):
  ● ETTm1  ● ETTm2  ● ETTh1  ● ETTh2  ● Weather  ● Traffic  ● Electricity

        ★ Solar        ← série nova → routing escolhe prompt mais próximo
        ★ Exchange       (ETT), que não representa bem esse domínio

  MoP com LOTSA+Chronos (~57 datasets):
  ● ETT  ● Weather  ● Traffic  ● Solar  ● Exchange  ● M4  ● Tourism ...
        → routing tem prompt adequado para cada região do espaço
```

**Argumento para o paper**: "nosso método melhora naturalmente com mais dados de pré-treino do MoP, sem retreinar o encoder" — diferencia do MOIRAI/Timer que precisam retreinar o modelo inteiro para cada novo corpus.

**Risco a verificar**: data leakage — se os 7 datasets ICML aparecem no LOTSA/Chronos, o LOO deixa de ser zero-shot real para esses datasets.
