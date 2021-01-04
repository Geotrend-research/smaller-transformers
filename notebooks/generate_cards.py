import os

for model_name in os.listdir('new-models'):
    text = open('README_sample.md').read()
    lang = model_name[10:len(model_name)-6]
    if len(lang)==2:
    	# mono-language models
        text = text.replace('language: ar', 'language: '+lang, 1)
    else:
    	# multi-language models
        text = text.replace('language: ar', 'language: multilingual', 1)
    text = text.replace('-ar-', '-'+lang+'-')
    fw = open(os.path.join('new-models', model_name, 'README.md'), 'w')
    fw.write(text)
    fw.close()