"""Assistente do Ofertão: responde dúvidas sobre o site usando um LLM.

Provedores (o primeiro que estiver configurado é usado):

| Variável de ambiente | Provedor | Custo |
|---|---|---|
| GEMINI_API_KEY | Google Gemini (aistudio.google.com) | plano gratuito |
| GROQ_API_KEY | Groq (console.groq.com), modelos abertos como Llama | plano gratuito |
| OFERTAO_CHAT_PROVEDOR=ollama | Ollama rodando no seu computador | grátis, local |
| ANTHROPIC_API_KEY | Claude (a mesma chave dos encartes) | pago |

Gemini, Groq e Ollama falam o formato da API da OpenAI (/chat/completions), então
um único cliente HTTP atende os três. Sem nenhum provedor, ou se ele falhar, o
assistente responde com o FAQ local (busca por palavras-chave), para o site nunca
ficar sem resposta.
"""

import logging
import os
import re
from dataclasses import dataclass

import httpx

from app.catalogo import normalizar

log = logging.getLogger(__name__)

MAX_MENSAGENS = 12
TIMEOUT_S = 30

BASE_CONHECIMENTO = """\
O Ofertão é um comparador gratuito de supermercados da Grande Natal (RN). O usuário monta a lista
de compras e o site mostra em qual supermercado REAL a compra completa sai mais barata, já somando
o gasto de ida e volta até a loja.

COMO USAR (página "Faça sua cotação", aberta pelo botão "Economize já" da página inicial):
1. Onde você está: digitar CEP, bairro ou cidade e tocar em "Buscar", ou tocar em "Usar minha
   localização". Também dá para tocar no mapa para ajustar a posição.
2. O que vai comprar: buscar o produto pelo nome (ex.: arroz, café, leite), filtrar pelas categorias
   (Arroz e feijão, Mercearia, Leite e queijo, Carnes, Limpeza, Higiene) ou tocar em "Usar cesta
   básica pronta", que coloca arroz, feijão, leite, açúcar, café e óleo na lista.
3. Sua lista: ajustar as quantidades com os botões − e +, ou "Limpar lista".
4. Tocar em "Ver onde é mais barato".
Em "Ajustes (opcional)" dá para mudar a distância máxima (padrão 15 km, de 1 a 100 km) e o gasto
com transporte por km (padrão R$ 0,80/km).

AGENTE DE IA (caixa "🤖 Agente de IA", no topo da página de cotação, acima de "Onde você está"):
É um atalho para fazer a cotação escrevendo o pedido do jeito que você falaria, sem montar a lista
item por item. Como usar:
1. (Recomendado) Antes, diga onde você está em "Onde você está" (CEP, bairro ou "Usar minha
   localização"): o agente usa essa posição. Sem ela, ele tenta achar um bairro ou cidade escrito no
   pedido (ex.: "no Alecrim", "em Parnamirim") e, se não achar, usa o centro de Natal.
2. Escrever o pedido na caixa de texto, por exemplo: "Quero 2 pacotes de arroz, feijão, café, leite e
   frango, só tenho 65 reais". Dá para dizer quantidades ("2 pacotes de arroz") e um limite de gasto.
   Também dá para tocar num dos exemplos prontos abaixo da caixa ("Exemplo com orçamento" ou
   "Cesta básica via IA") para preencher o texto.
3. (Opcional) Digitar um limite de gasto no campo "Orçamento R$" (ou escrever no próprio pedido,
   como "tenho 65 reais").
4. Tocar em "✨ Perguntar ao Agente" e aguardar ("Agente raciocinando...").
O agente responde com um resumo e mostra a mesma tela de resultado da cotação normal (loja mais
barata, total, mapa e rota). Abaixo da resposta aparecem as "Ferramentas acionadas pelo Agente", os
passos que ele seguiu: achar a localização, encontrar os produtos no banco de dados e calcular a
melhor compra. Se a compra passar do orçamento, ele avisa e sugere tirar o item mais caro para caber
no valor, mostrando o novo total.
Dicas: use nomes simples de produtos da feira (arroz, feijão, café, açúcar, óleo, leite, macarrão,
molho, frango, ovos, manteiga, margarina, pão, biscoito, flocão/cuscuz, farinha, sal, sardinha,
detergente, sabão, água sanitária, papel higiênico, creme dental, sabonete). Se nenhum produto for
reconhecido, ele monta uma cesta básica padrão. Se não aparecer loja, aumente a distância em
"Ajustes (opcional)". A distância máxima configurada em Ajustes também vale para o agente.
Diferença: este assistente de dúvidas (o chat) só explica como o site funciona; quem calcula preços a
partir de um pedido escrito é o Agente de IA da página de cotação.

O RESULTADO mostra: a loja mais barata, o total (produtos + transporte), quanto você economiza em
relação à opção mais cara, o preço de cada produto ("Ver preço de cada produto") e as outras opções
em ordem. O mapa mostra as lojas e a rota até a escolhida. Quando passar em DUAS lojas próximas
economiza pelo menos R$ 3,00, o site sugere dividir a feira e mostra o que comprar em cada uma.

COMO O CÁLCULO FUNCIONA:
- custo total = soma de (preço × quantidade) + distância de ida e volta × R$ por km.
- Item genérico (ex.: "Arroz branco tipo 1 5kg"): usa o produto mais barato daquele tipo em cada loja,
  de qualquer marca. Para uma marca específica, basta buscar pelo nome dela.
- Só entram lojas com preço para pelo menos um item da lista, e só preços de até 30 dias.
- Se um item falta numa loja, ele é estimado pelo preço do meio (mediana) das outras lojas + 20%,
  só para a comparação ficar justa. Os itens faltantes aparecem na loja.

DE ONDE VÊM OS DADOS:
- Lojas: OpenStreetMap, em Natal, Parnamirim, São Gonçalo do Amarante, Macaíba, Extremoz,
  Ceará-Mirim, São José de Mipibu e Nísia Floresta.
- Preços da loja online do Atacadão (podem diferir da prateleira).
- Encartes de ofertas das redes, lidos por IA, com validade; saem da conta quando vencem.
- Cupons fiscais (NFC-e) enviados pelos usuários, lidos por IA a partir da foto.
- Preços informados pelos donos das lojas, num painel protegido por senha.
Carrefour e Nordestão/Favorito ainda não têm preços automáticos (bloqueiam acesso ou exigem parceria).

OUTRAS INFORMAÇÕES:
- É grátis e não precisa de cadastro nem login.
- Não vende nada e não faz entregas: só compara preços. A compra é feita na loja.
- Preços são uma estimativa com base nos dados coletados; podem mudar na loja.
- Donos de mercado podem enviar seus preços pelo painel do dono (falar com a equipe para receber
  o token de acesso).
"""

INSTRUCOES = """\
Você é o assistente virtual do Ofertão. Responda em português do Brasil, de forma curta (no máximo
uns 5 parágrafos curtos ou uma lista), amigável e direta, usando SOMENTE as informações abaixo.
Se a pergunta não for sobre o Ofertão, compras de supermercado ou economia nas compras, diga com
educação que só pode ajudar com o Ofertão. Se a informação não estiver abaixo, diga que não sabe
em vez de inventar. Nunca invente preços, lojas ou números: para saber preços, oriente o usuário a
fazer a cotação (montar a lista e tocar em "Ver onde é mais barato"). Refira-se às páginas pelo nome
("página de cotação", "página inicial"), nunca por endereço ou link. Não use tabelas nem títulos em
markdown; negrito com **texto** é ok.

INFORMAÇÕES SOBRE O SITE:
{base}
DADOS DE AGORA: {status}
"""


@dataclass(frozen=True)
class Provedor:
    nome: str
    url: str
    modelos: tuple[str, ...]  # tentados em ordem: se um estiver sobrecarregado/indisponível, vai o próximo
    chave: str | None


# Apelidos "-latest" acompanham a versão atual do Google (versões antigas são desligadas para contas novas).
# O "lite" vem primeiro: responde em ~1 s e quase nunca fica sobrecarregado no plano gratuito.
MODELOS_GEMINI = ("gemini-flash-lite-latest", "gemini-flash-latest")
# Erros em que vale tentar o próximo modelo: não existe, cota/limite, sobrecarga.
STATUS_TENTAR_OUTRO = {404, 429, 500, 502, 503, 504}


def provedor_configurado() -> Provedor | None:
    """Escolhe o provedor pelo que estiver nas variáveis de ambiente."""
    modelo = os.getenv("OFERTAO_CHAT_MODELO")
    forcado = os.getenv("OFERTAO_CHAT_PROVEDOR", "").strip().lower()

    def m(*padrao: str) -> tuple[str, ...]:
        return (modelo,) if modelo else padrao

    if forcado == "ollama":
        base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        return Provedor("ollama", f"{base}/v1/chat/completions", m("llama3.2"), None)
    if forcado == "nenhum":
        return None
    if (forcado in ("", "gemini")) and (chave := os.getenv("GEMINI_API_KEY")):
        return Provedor("gemini",
                        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                        m(*MODELOS_GEMINI), chave)
    if (forcado in ("", "groq")) and (chave := os.getenv("GROQ_API_KEY")):
        return Provedor("groq", "https://api.groq.com/openai/v1/chat/completions",
                        m("llama-3.3-70b-versatile"), chave)
    if (forcado in ("", "anthropic")) and (chave := os.getenv("ANTHROPIC_API_KEY")):
        return Provedor("anthropic", "", m("claude-haiku-5-5"), chave)
    return None


def _prompt_sistema(status: dict | None) -> str:
    if status:
        fontes = ", ".join(f"{k}: {v}" for k, v in (status.get("precos_por_fonte") or {}).items())
        texto = (f"{status.get('lojas', 0)} lojas cadastradas, {status.get('lojas_com_preco', 0)} com preços, "
                 f"{status.get('produtos', 0)} produtos. Preços por fonte: {fontes or 'nenhum ainda'}.")
    else:
        texto = "indisponível."
    return INSTRUCOES.format(base=BASE_CONHECIMENTO, status=texto)


def _chamar_openai_compat(p: Provedor, sistema: str, mensagens: list[dict],
                          cliente: httpx.Client) -> str:
    cabecalhos = {"Authorization": f"Bearer {p.chave}"} if p.chave else {}
    erro: Exception | None = None
    for modelo in p.modelos:
        corpo = {
            "model": modelo,
            "messages": [{"role": "system", "content": sistema}, *mensagens],
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        try:
            r = cliente.post(p.url, json=corpo, headers=cabecalhos, timeout=TIMEOUT_S)
            r.raise_for_status()
            texto = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if texto:
                return texto
            erro = ValueError(f"{modelo}: resposta vazia")
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in STATUS_TENTAR_OUTRO:
                raise  # chave inválida (400/401/403): trocar de modelo não resolve
            erro = e
        except httpx.TimeoutException as e:
            erro = e
        log.info("Chat: modelo %s indisponível (%s); tentando o próximo", modelo, erro)
    raise erro or RuntimeError("nenhum modelo configurado")


def _chamar_anthropic(p: Provedor, sistema: str, mensagens: list[dict]) -> str:
    import anthropic

    resp = anthropic.Anthropic(api_key=p.chave).messages.create(
        model=p.modelos[0], max_tokens=600, system=sistema, messages=mensagens,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ---------- FAQ local (sem IA externa) ----------

FAQ: list[tuple[tuple[str, ...], str]] = [
    (("agente", "o agente", "agente de ia", "usar o agente", "inteligencia artificial", "robo", "escrever o pedido", "escrevendo", "pedido escrito",
      "orcamento", "so tenho", "perguntar ao agente"),
     "O **Agente de IA** fica no topo da página de cotação (caixa \"🤖 Agente de IA\"). Como usar: "
     "**1)** diga onde você está em \"Onde você está\" (ou escreva o bairro no pedido); "
     "**2)** escreva o pedido normalmente, ex.: \"Quero 2 pacotes de arroz, feijão, café e leite, só tenho 65 reais\", "
     "ou toque num dos exemplos prontos; **3)** se quiser, coloque um limite em \"Orçamento R$\"; "
     "**4)** toque em \"✨ Perguntar ao Agente\". Ele encontra os produtos, calcula a loja mais barata com o "
     "transporte e, se passar do orçamento, sugere tirar o item mais caro. Os passos que ele seguiu aparecem "
     "em \"Ferramentas acionadas pelo Agente\"."),
    (("gratis", "gratuito", "pagar", "pago", "custa", "cadastro", "criar conta", "login"),
     "O Ofertão é **totalmente gratuito** e não precisa de cadastro: é só abrir a página de cotação e usar."),
    (("como usar", "como funciona", "comecar", "passo", "cotacao", "usar o site"),
     "Saiba como usar em 3 passos: **1)** diga onde você está (CEP, bairro ou \"Usar minha localização\"); "
     "**2)** escolha os produtos ou toque em \"Usar cesta básica pronta\"; "
     "**3)** toque em \"Ver onde é mais barato\". O site mostra a loja com o menor total, já com a ida e a volta."),
    (("calculo", "calcula", "conta", "transporte", "deslocamento", "km", "gasolina", "distancia"),
     "O total é a soma dos produtos mais o custo de ida e volta até a loja (distância × R$ por km, "
     "padrão R$ 0,80/km). Você muda a distância máxima e o custo por km em **Ajustes (opcional)**."),
    (("de onde", "fonte", "dados", "precos vem", "confiavel", "atualiza", "atualizado"),
     "Os preços vêm da loja online do Atacadão, de encartes de ofertas das redes (lidos por IA), de cupons "
     "fiscais enviados por usuários e de preços informados pelos donos das lojas. Só usamos preços de até 30 dias. "
     "As lojas vêm do OpenStreetMap."),
    (("cidade", "natal", "parnamirim", "regiao", "onde funciona", "bairro", "cep"),
     "O Ofertão cobre a Grande Natal: Natal, Parnamirim, São Gonçalo do Amarante, Macaíba, Extremoz, "
     "Ceará-Mirim, São José de Mipibu e Nísia Floresta."),
    (("dividir", "duas lojas", "dois mercados", "divisao"),
     "Quando comprar em **duas lojas próximas** economiza pelo menos R$ 3,00 (já contando a rota), "
     "o site sugere dividir a feira e mostra o que comprar em cada uma."),
    (("marca", "generico", "especifico"),
     "Item genérico (ex.: \"Arroz tipo 1 5kg\") usa o mais barato daquele tipo, de qualquer marca. "
     "Se você faz questão de uma marca, busque pelo nome dela e adicione o produto específico."),
    (("falta", "faltante", "nao tem", "sem preco", "nenhum mercado"),
     "Se uma loja não tem preço para algum item, ele aparece como faltante e, só para comparar de forma justa, "
     "é estimado pela mediana das outras lojas + 20%. Se não aparecer nenhuma loja, aumente a distância em Ajustes."),
    (("carrefour", "nordestao", "favorito"),
     "Carrefour e Nordestão/Favorito ainda não têm preços automáticos: um bloqueia acesso automatizado e o "
     "outro exige parceria. Eles podem entrar por cupons fiscais ou pelo painel do dono."),
    (("cupom", "nota fiscal", "nfc"),
     "Os preços de cupons fiscais vêm de fotos enviadas pelos usuários; uma IA lê os produtos e os preços pagos."),
    (("encarte", "oferta", "promocao"),
     "As ofertas dos encartes das redes são lidas por IA, valem para todas as lojas da rede e saem da conta quando vencem."),
    (("dono", "lojista", "meu mercado", "cadastrar loja"),
     "Donos de mercado podem enviar os preços da loja pelo painel do dono. Fale com a equipe do Ofertão para receber o acesso."),
    (("entrega", "comprar pelo site", "delivery", "pedido"),
     "O Ofertão não vende nem entrega: ele só compara os preços. A compra é feita direto na loja."),
]

RESPOSTA_PADRAO = (
    "Posso ajudar com dúvidas sobre o Ofertão: como fazer a cotação, como usar o Agente de IA, como o "
    "cálculo funciona, de onde vêm os preços e quais cidades são atendidas. O que você quer saber?"
)


def responder_faq(pergunta: str) -> str:
    texto = normalizar(pergunta)
    melhor, pontos = RESPOSTA_PADRAO, 0
    for chaves, resposta in FAQ:
        p = sum(len(c) for c in chaves
                if re.search(rf"(?<![a-z0-9]){re.escape(normalizar(c))}", texto))
        if p > pontos:
            melhor, pontos = resposta, p
    return melhor


# ---------- ponto de entrada ----------

def limpar_historico(mensagens: list[dict]) -> list[dict]:
    """Mantém as últimas mensagens, alternando usuário/assistente e terminando no usuário."""
    limpas = [{"role": m["role"], "content": m["content"].strip()[:1500]}
              for m in mensagens if m["role"] in ("user", "assistant") and m["content"].strip()]
    limpas = limpas[-MAX_MENSAGENS:]
    while limpas and limpas[0]["role"] != "user":
        limpas.pop(0)
    return limpas


def responder(mensagens: list[dict], status: dict | None = None,
              cliente: httpx.Client | None = None) -> tuple[str, str]:
    """Devolve (resposta, provedor usado)."""
    historico = limpar_historico(mensagens)
    if not historico or historico[-1]["role"] != "user":
        return RESPOSTA_PADRAO, "faq"
    pergunta = historico[-1]["content"]

    p = provedor_configurado()
    if p is None:
        return responder_faq(pergunta), "faq"

    sistema = _prompt_sistema(status)
    try:
        if p.nome == "anthropic":
            texto = _chamar_anthropic(p, sistema, historico)
        else:
            with (cliente or httpx.Client()) as c:
                texto = _chamar_openai_compat(p, sistema, historico, c)
        if texto:
            return texto, p.nome
    except Exception as e:  # provedor fora do ar, cota gratuita esgotada, chave inválida...
        log.warning("Chat: falha no provedor %s (%s); usando FAQ local", p.nome, e)
    return responder_faq(pergunta), "faq"
