import random


from pytorch_translation.data import prepare_data


input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
print(random.choice(pairs))
