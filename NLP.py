from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration, MBartForConditionalGeneration, MBart50TokenizerFast
from rouge import Rouge
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

import time
import numpy as np

def divir_em_blocos_20(vetor, tamanho):
    return [vetor[i:i + tamanho] for i in range(0, len(vetor), tamanho)]

def join_blocos_em_string(vetor_dividido):
    sentenceGroups = []
    for i, v in enumerate(vetor_dividido):
        chunk = ' '.join(v)
        sentenceGroups.append(chunk)
    return sentenceGroups

def calcMeteor(reference, candidate):
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = word_tokenize(reference)
    meteorScore = meteor_score([reference_tokens], candidate_tokens)
    return meteorScore

def calcBleu(reference, candidate):
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = word_tokenize(reference)
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    return bleu_score

def calcRouge(references, candidates):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidates, references, avg=True)
    return rouge_scores

def calcMeteorChunks():
    scores = []
    for i in range(len(chunks_translated_sentences)):
        meteor = calcMeteor(chunks_reference_translations[i], chunks_translated_sentences[i])
        scores.append(meteor)
        output_file.write(f"Bloco {i+1} - Meteor Score: {meteor}\n")
    scoreMeteorAvg = np.average(scores)
    return scoreMeteorAvg

def calcBleuChunks():
    scores = []
    for i in range(len(chunks_translated_sentences)):
        bleu = calcBleu(chunks_reference_translations[i], chunks_translated_sentences[i])
        scores.append(bleu)
        output_file.write(f"Bloco {i+1} - BLEU Score: {bleu}\n")
    scoreBleuAvg = np.average(scores)
    return scoreBleuAvg

def generate_translation(sentence, reference):
    output_file.write(f"Input: {sentence}\n")
    output_file.write(f"Target: {reference}\n")

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=5)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    output_file.write(f"Prediction: {prediction}\n")

    return prediction

def translate(sentences):
    translated_sentences = []
    i = 1
    for sentence, reference in zip(sentences, reference_translations):
        output_file.write(f"Rodada {i}\n")
        i += 1
        prediction = generate_translation(sentence, reference)
        translated_sentences.append(prediction)
    return translated_sentences

if __name__ == '__main__':
    with open('results.txt', 'w', encoding='utf-8') as output_file:
        # Seu código aqui
        with open('pln/19_10pt.txt', 'r', encoding='utf-8') as fPt:
            reference_translations = [line.strip() for line in fPt if line.strip()]

        with open('pln/19_10en.txt', 'r', encoding='utf-8') as fEn:
            sentences = [line.strip() for line in fEn if line.strip()]

        sentences_divididos = divir_em_blocos_20(sentences, 20)
        reference_translations_divididos = divir_em_blocos_20(reference_translations, 20)
        chunks_sentences = join_blocos_em_string(sentences_divididos)
        chunks_reference_translations = join_blocos_em_string(reference_translations_divididos)

        results = {
            "opus-mt-en-ROMANCE": {
                "exec_time": None,
                "meteor": None,
                "bleu:": None
            },
            'opus-mt-tc-big-en-pt': {
                "exec_time": None,
                "meteor": None,
                "bleu:": None
            },
            'translation-en-pt-t5': {
                "exec_time": None,
                "meteor": None,
                "bleu:": None
            },
            'mbart-large-50-many-to-many-mmt': {
                "exec_time": None,
                "meteor": None,
                "bleu": None
            }
        }

        def translateBart(sentences):
            i = 1
            tokenizer.src_lang = "en_XX"
            translated_sentences = []
            for sentence, reference in zip(sentences, reference_translations):
                encoded_en = tokenizer(sentence, return_tensors="pt")
                output_file.write(f"Rodada {i}\n")
                i += 1
                output_file.write(f"Input: {sentence}\n")
                output_file.write(f"Target: {reference}\n")
                generated_tokens = model.generate(
                    **encoded_en,
                    forced_bos_token_id=tokenizer.lang_code_to_id["pt_XX"]
                )
                prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                output_file.write(f"Prediction: {prediction[0]}\n")
                translated_sentences.append(prediction[0])
            return translated_sentences

        ##########################################################################################

        output_file.write('Marian MT (Helsinki-NLP/opus-mt-en-ROMANCE)\n')
        output_file.write('40 sentences\n')
        model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        prefixedSentences = ['>>pt_BR<< ' + sentence for sentence in sentences]

        start_time = time.time()
        translated_sentences = translate(prefixedSentences)
        end_time = time.time()
        exec_time = end_time - start_time
        output_file.write(f"Tempo de execução: {exec_time} segundos\n")

        translated_sentences_divididos = divir_em_blocos_20(translated_sentences, 20)
        chunks_translated_sentences = join_blocos_em_string(translated_sentences_divididos)

        meteor = calcMeteorChunks()
        output_file.write(f"Meteor: {meteor}\n")

        bleu = calcBleuChunks()
        output_file.write(f"BLEU: {bleu}\n")

        results['opus-mt-en-ROMANCE']['exec_time'] = exec_time
        results['opus-mt-en-ROMANCE']['meteor'] = meteor
        results['opus-mt-en-ROMANCE']['bleu'] = bleu


        

        ##########################################################################################

        output_file.write('Marian MT (Helsinki-NLP/opus-mt-tc-big-en-pt)\n')
        output_file.write('40 sentences\n')
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        start_time = time.time()
        translated_sentences = translate(sentences)
        end_time = time.time()
        exec_time = end_time - start_time
        output_file.write(f"Tempo de execução: {exec_time} segundos\n")

        translated_sentences_divididos = divir_em_blocos_20(translated_sentences, 20)
        chunks_translated_sentences = join_blocos_em_string(translated_sentences_divididos)

        meteor = calcMeteorChunks()
        output_file.write(f"Meteor: {meteor}\n")

        bleu = calcBleuChunks()
        output_file.write(f"BLEU: {bleu}\n")

        results['opus-mt-tc-big-en-pt']['exec_time'] = exec_time
        results['opus-mt-tc-big-en-pt']['meteor'] = meteor
        results['opus-mt-tc-big-en-pt']['bleu'] = bleu

        ##########################################################################################

        output_file.write('T5 (unicamp-dl/translation-en-pt-t5)\n')
        output_file.write('40 sentences\n')
        model_name = "unicamp-dl/translation-en-pt-t5"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

        model = T5ForConditionalGeneration.from_pretrained(model_name)

        prefixedSentences = ['translate English to Portuguese: ' + sentence for sentence in sentences]

        start_time = time.time()
        translated_sentences = translate(prefixedSentences)
        end_time = time.time()
        exec_time = end_time - start_time
        output_file.write(f"Tempo de execução: {exec_time} segundos\n")

        translated_sentences_divididos = divir_em_blocos_20(translated_sentences, 20)
        chunks_translated_sentences = join_blocos_em_string(translated_sentences_divididos)

        meteor = calcMeteorChunks()
        output_file.write(f"Meteor: {meteor}\n")

        bleu = calcBleuChunks()
        output_file.write(f"BLEU: {bleu}\n")

        results['translation-en-pt-t5']['exec_time'] = exec_time
        results['translation-en-pt-t5']['meteor'] = meteor
        results['translation-en-pt-t5']['bleu'] = bleu

        ##########################################################################################

        output_file.write('mBart (facebook/mbart-large-50-many-to-many-mmt)\n')
        output_file.write('40 sentences\n')
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        start_time = time.time()

        translated_sentences = translateBart(sentences)

        end_time = time.time()
        exec_time = end_time - start_time
        output_file.write(f"Tempo de execução: {exec_time} segundos\n")

        translated_sentences_divididos = divir_em_blocos_20(translated_sentences, 20)
        chunks_translated_sentences = join_blocos_em_string(translated_sentences_divididos)

        meteor = calcMeteorChunks()
        output_file.write(f"Meteor: {meteor}\n")

        bleu = calcBleuChunks()
        output_file.write(f"BLEU: {bleu}\n")

        results['mbart-large-50-many-to-many-mmt']['exec_time'] = exec_time
        results['mbart-large-50-many-to-many-mmt']['meteor'] = meteor
        results['mbart-large-50-many-to-many-mmt']['bleu'] = bleu

        ##########################################################################################

        for element in results:
            output_file.write(f"{element}\n")
        for key, value in results[element].items():
            output_file.write(f"    {key}: {value}\n")
        
        rouge_avg = calcRouge(reference_translations, sentences)
        output_file.write(f"Medias gerais ROUGE: {rouge_avg}")
