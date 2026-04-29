**In-context learning (ICL)** é a capacidade do modelo de "aprender" algo novo **durante a inferência** — ou seja, só de ler o prompt — sem nunca ter seus pesos atualizados.

---

## A diferença crucial

| Treino tradicional | In-context learning |
|---|---|
| Ajusta bilhões de pesos com gradient descent | Nenhum peso muda. O modelo só lê o prompt e responde. |
| Demora horas/dias | Acontece em milissegundos |
| Exemplo: fine-tuning em dataset de tradução | Exemplo: você cola 3 exemplos de tradução no prompt e ele traduz o 4º |

---

## O exemplo clássico (few-shot)

Você dá pro modelo:

```
Português: O gato está no telhado.
Inglês: The cat is on the roof.

Português: A maçã é vermelha.
Inglês: The apple is red.

Português: O cachorro late.
Inglês:
```

O modelo nunca foi treinado especificamente pra traduzir **"O cachorro late"**. Mas vendo o padrão A→B nos exemplos anteriores do **mesmo prompt**, ele generaliza e completa com:

```
Inglês: The dog barks.
```

Isso é ICL. Ele "aprendeu" a tarefa de tradução **só do contexto** que você colou.

---

## E o que isso tem a ver com Induction Heads?

Induction heads são um mecanismo mecanicista que **implementa** uma forma específica de ICL.

O texto do notebook descreve exatamente isso:

> *"generalising from one observation that token B follows token A, to predict that token B will follow A in future occurrences of A"*

Traduzindo:

1. No prompt aparece: `... A B ... A ...`
2. O modelo viu uma única vez que `A` foi seguido por `B`.
3. Quando `A` aparece de novo mais tarde no **mesmo prompt**, o induction head copia o padrão e prediz `B`.

Ou seja: o modelo não memorizou `A→B` durante o treino. Ele **detectou o padrão no contexto atual** e aplicou. É ICL em nível de circuito.

---

## Resumo em uma frase

> **In-context learning** = o modelo adapta seu comportamento só de ler exemplos no prompt, sem treino. **Induction heads** = o circuito interno que permite isso acontecer para padrões do tipo "A seguido de B".

No notebook, quando você for fazer activation patching em prompts do tipo `A B C D A B`, você vai ver exatamente esse circuito copiando o padrão. É a prova mecanicista de que ICL não é mágica — é computação localizada em cabeças de atenção específicas.