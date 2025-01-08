import tkinter as tk
from tkinter import messagebox, filedialog
import csv
import os

# Função para validar os arquivos

def validar_arquivos(arquivos):
    dados = {}
    for arquivo in arquivos:
        if not os.path.isfile(arquivo):
            messagebox.showerror("Erro", f"Arquivo não encontrado: {arquivo}")
            return None

        with open(arquivo, "r", encoding="utf-8") as f:
            linhas = f.readlines()

        if len(linhas) < 2 or not linhas[0].strip():
            messagebox.showerror("Erro", f"Formato inválido no arquivo: {arquivo}")
            return None

        modelo = linhas[0].strip()
        traducoes = []

        for i in range(1, len(linhas)):
            linha = linhas[i].strip()
            if linha.startswith("Rodada "):
                if i + 1 >= len(linhas) or not linhas[i + 1].strip():
                    messagebox.showerror(
                        "Erro", f"Rodada sem tradução encontrada no arquivo: {arquivo}, linha {i + 1}"
                    )
                    return None
                traducoes.append(linhas[i + 1].strip())

        if len(traducoes) != 143:
            messagebox.showerror(
                "Erro", f"O arquivo {arquivo} contém {len(traducoes)} rodadas em vez de 143."
            )
            return None

        dados[modelo] = traducoes

    return dados

# Função para salvar resultados em CSV
def salvar_resultados(resultados, nome_arquivo):
    with open(nome_arquivo, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Rodada", "Modelos Escolhidos"])
        for rodada, modelos in enumerate(resultados, 1):
            writer.writerow([rodada, ", ".join(modelos)])

# Função principal para exibição da interface gráfica
def exibir_interface(dados):
    rodada_atual = [1]
    escolhas = []
    modelos = list(dados.keys())

    def atualizar_interface():
        for i, modelo in enumerate(modelos):
            botoes_check[i].config(text=dados[modelo][rodada_atual[0] - 1])

    def proxima_rodada():
        selecoes = [modelos[i] for i in range(len(modelos)) if variaveis_opcoes[i].get() == 1]
        if not selecoes:
            messagebox.showerror("Erro", "Por favor, selecione pelo menos uma tradução antes de continuar.")
            return

        escolhas.append(selecoes)
        rodada_atual[0] += 1

        if rodada_atual[0] > 143:
            salvar_resultados(escolhas, "resultados.csv")
            resultados_finais = {modelo: sum([1 for rodada in escolhas if modelo in rodada]) for modelo in modelos}
            mensagem = "\n".join(
                f"{modelo}: {score}/143" for modelo, score in resultados_finais.items()
            )
            messagebox.showinfo("Resultados", mensagem)
            root.destroy()
        else:
            atualizar_interface()
            for variavel in variaveis_opcoes:
                variavel.set(0)

    root = tk.Tk()
    root.title("Avaliação de Traduções")

    frame = tk.Frame(root)
    frame.pack(pady=20, padx=20)

    variaveis_opcoes = [tk.IntVar(value=0) for _ in modelos]
    botoes_check = []

    for i in range(len(modelos)):
        botao = tk.Checkbutton(
            frame, text="", variable=variaveis_opcoes[i], anchor="w", justify="left"
        )
        botao.pack(fill="x", padx=10, pady=5)
        botoes_check.append(botao)

    botao_proximo = tk.Button(
        root, text="Próxima", command=proxima_rodada, width=15, height=2
    )
    botao_proximo.pack(pady=20)

    atualizar_interface()
    root.mainloop()

# Seleção dos arquivos
if __name__ == "__main__":
    arquivos_selecionados = filedialog.askopenfilenames(
        title="Selecione os arquivos de tradução",
        filetypes=[("Arquivos de Texto", "*.txt")],
    )

    if arquivos_selecionados:
        dados_validos = validar_arquivos(arquivos_selecionados)
        if dados_validos:
            exibir_interface(dados_validos)
    else:
        messagebox.showinfo("Informação", "Nenhum arquivo selecionado.")
