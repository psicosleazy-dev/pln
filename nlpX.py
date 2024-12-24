from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration, MBartForConditionalGeneration, MBart50Tokenizer
import time
import numpy as np
from collections import defaultdict
from tabulate import tabulate

if __name__ == '__main__':
    # Load input data
    with open('19_10pt.txt', 'r', encoding='utf-8') as fPt:
        reference_translations = [line.strip() for line in fPt if line.strip()]
    with open('19_10en.txt', 'r', encoding='utf-8') as fEn:
        sentences = [line.strip() for line in fEn if line.strip()]


    # Constants
    TIPO_MAPPING = {1: 50, 2: 300, 3:500}
    tgt_lang_id = None
    MODELS = [
        {"name": "Helsinki-NLP/opus-mt-en-ROMANCE", "type": "marian", "prefix": ">>pt_BR<< "},
        {"name": "Helsinki-NLP/opus-mt-tc-big-en-pt", "type": "marian", "prefix": None},
        {"name": "unicamp-dl/translation-en-pt-t5", "type": "t5", "prefix": "translate English to Portuguese: "},
        {"name": "facebook/mbart-large-50-many-to-many-mmt", "type": "mbart", "src_lang": "en_XX", "tgt_lang": "pt_XX"}
    ]
    ROUNDS = 5

    # Utility functions
    def tipo_to_number_sentences(tipo):
        return TIPO_MAPPING.get(tipo, None)

    def choose_sentences(k,sentences):
        target_word_count = tipo_to_number_sentences(k + 1)
        word_count = 0
        sentences_up_to_target = []

        for i, sentence in enumerate(sentences[3:]):
            words_in_sentence = sentence.split()
            word_count+=len(words_in_sentence) 
            if word_count >= target_word_count:
                word_count-=len(words_in_sentence)
                if(k==2):
                    
                    word_count += len(words_in_sentence)
                    sentences_up_to_target.append(sentence)
                break
            sentences_up_to_target.append(sentence)        
        print(f"Tipo {k+1} - {word_count} palavras - {i} sentenças.")
        return sentences_up_to_target,word_count

    def generate_translation(sentence, model, tokenizer, model_type, **kwargs):
        # print(sentence)
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if model_type == "mbart":
            outputs = model.generate(**inputs, forced_bos_token_id=kwargs['tgt_lang'])
        else:
            outputs = model.generate(**inputs, max_new_tokens=512, num_beams=5)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def measure_execution_time(sentences, model, tokenizer, model_type, rounds, prefix=None, **kwargs):

        exec_times = []
        for round in range(rounds):
            start_time = time.time()
            translated_sentences = [
                generate_translation((prefix or "") + sentence, model, tokenizer, model_type, **kwargs)
                for sentence in sentences
            ]
            # print(translated_sentences)
            end_time = time.time()
            exec_time = end_time - start_time
            exec_times.append(exec_time)
            print(f"Modelo {model_name}, Tipo {k + 1}, Rodada {round + 1}: {exec_time:.4f} segundos.")
        return np.mean(exec_times), np.std(exec_times), exec_times

    # Main execution

    models_info = defaultdict(lambda: defaultdict(dict))  # Estrutura para armazenar os dados

    for k in range(3):
        
        sentences_up_to_target,word_count = choose_sentences(k,sentences)

        for model_config in MODELS:
            model_name = model_config["name"]
            print(f"Testando modelo: {model_name}")

            # Load model and tokenizer
            if model_config["type"] == "marian":
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
            elif model_config["type"] == "t5":
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            elif model_config["type"] == "mbart":
                tokenizer = MBart50Tokenizer.from_pretrained(model_name)
                model = MBartForConditionalGeneration.from_pretrained(model_name)
                tokenizer.src_lang = model_config["src_lang"]
                tgt_lang_id = tokenizer.lang_code_to_id.get(model_config.get("tgt_lang"))
                            

            # Warm-up
            generate_translation(
                "Warmup sentence", 
                model, 
                tokenizer, 
                model_config["type"], 
                tgt_lang=tgt_lang_id
            )

            # Measure execution time
            mean_time, std_time, exec_times = measure_execution_time(
                sentences_up_to_target,
                model,
                tokenizer,
                model_config["type"],
                ROUNDS,
                prefix=model_config.get("prefix"),
                tgt_lang=tgt_lang_id
            )

            

            print(f"{model_name} - Tipo {k + 1}: Média = {mean_time:.2f}s, Desvio = {std_time:.4f}s")

            models_info[model_name][f"Tipo {k + 1} - {word_count} palavras"] = {
                "exec_times": exec_times,
                "mean_time": mean_time,
                "std_time": std_time
            }

    for model, results in models_info.items():
        print(f"\nResultados para o modelo {model}:")
        for tipo, stats in results.items():
            print(f"  {tipo}:")
            print(f"    Tempos de execução: {stats['exec_times']}")
            print(f"    Média: {stats['mean_time']:.4f}s")
            print(f"    Desvio padrão: {stats['std_time']:.4f}s")

    table_data = []

    for model, results in models_info.items():
        for tipo, stats in results.items():
            table_data.append([
                model,
                tipo,
                stats["exec_times"],
                f"{stats['mean_time']:.4f}s",
                f"{stats['std_time']:.4f}s"
            ])

    # Definir cabeçalhos
    headers = ["Modelo", "Tipo", "Tempos de Execução", "Média", "Desvio Padrão"]

    # Exibir a tabela em formato grid
    print(tabulate(table_data, headers=headers, tablefmt="grid"))