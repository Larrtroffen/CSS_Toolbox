import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords
# nltk.download('punkt') # Run these once if not downloaded
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

import spacy
# python -m spacy download en_core_web_sm # Run once in terminal to download model
# nlp_spacy = spacy.load('en_core_web_sm')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "text_analysis_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 资源管理 (Resource Management - NLTK, spaCy)
# ==============================================================================

def ensure_nltk_resources(resources=None):
    """
    确保必要的NLTK资源已下载。

    参数:
    resources (list of str, optional): 需要检查的NLTK资源列表。
        例如: ['punkt', 'averaged_perceptron_tagger', 'stopwords']。
        如果为None，则检查默认的一组常用资源。
    """
    if resources is None:
        resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']

    resource_map = {
        'punkt': lambda: word_tokenize("test text for punkt"),
        'averaged_perceptron_tagger': lambda: pos_tag(word_tokenize("test text for tagger")),
        'stopwords': lambda: stopwords.words('english')
    }

    for res_name in resources:
        try:
            if res_name in resource_map:
                resource_map[res_name]()
            else:
                # 对于未在map中直接测试的资源，尝试加载
                nltk.data.find(f'corpora/{res_name}' if res_name == 'stopwords' else f'tokenizers/{res_name}' if res_name == 'punkt' else f'taggers/{res_name}')
            # print(f"NLTK资源 '{res_name}' 已存在。")
        except LookupError:
            print(f"NLTK资源 '{res_name}' 未找到。正在下载...")
            try:
                nltk.download(res_name, quiet=False)
                print(f"NLTK资源 '{res_name}' 下载成功。")
            except Exception as e:
                print(f"下载NLTK资源 '{res_name}' 失败: {e}")
        except Exception as e:
            print(f"检查NLTK资源 '{res_name}' 时发生未知错误: {e}")

def load_spacy_model(model_name='en_core_web_sm'):
    """
    加载spaCy模型，如果模型未找到，则尝试下载。

    参数:
    model_name (str, optional): 要加载的spaCy模型名称。
                                默认为 'en_core_web_sm'。

    返回:
    spacy.lang: 加载的spaCy语言模型对象，如果失败则返回None。
    """
    try:
        nlp = spacy.load(model_name)
        print(f"spaCy模型 '{model_name}' 加载成功。")
        return nlp
    except OSError:
        print(f"spaCy模型 '{model_name}' 未找到。正在尝试下载...")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            print(f"spaCy模型 '{model_name}' 下载并加载成功。")
            return nlp
        except Exception as e:
            print(f"下载或加载spaCy模型 '{model_name}' 失败: {e}")
            print(f"请尝试手动安装: python -m spacy download {model_name}")
            return None

# ==============================================================================
# NLTK 处理函数 (NLTK Processing Functions)
# ==============================================================================

def nltk_tokenize_text(text, language='english', lower=True):
    """
    使用NLTK对文本进行分词。

    参数:
    text (str): 要分词的输入文本。
    language (str, optional): 文本语言，用于分词器。默认为'english'。
    lower (bool, optional): 是否在分词前将文本转换为小写。默认为True。

    返回:
    list of str: 分词后的词语列表。
    """
    ensure_nltk_resources(['punkt'])
    if lower:
        text = text.lower()
    return word_tokenize(text, language=language)

def nltk_pos_tag_tokens(tokens):
    """
    使用NLTK对词语列表进行词性标注。

    参数:
    tokens (list of str): 词语列表。

    返回:
    list of tuples: 词性标注结果，每个元组为 (词语, 词性标签)。
    """
    ensure_nltk_resources(['averaged_perceptron_tagger'])
    return pos_tag(tokens)

def nltk_freq_dist(tokens, num_most_common=20, plot_title='词频分布图', save_filename_png=None):
    """
    计算词语列表的频率分布，并可选择绘制和保存图表。

    参数:
    tokens (list of str): 词语列表。
    num_most_common (int, optional): 要显示的最高频词数量。默认为20。
    plot_title (str, optional): 频率分布图的标题。
    save_filename_png (str, optional): 频率分布图的保存文件名 (例如 'freq_dist.png')。
                                     如果提供，则保存图表；否则不保存 (仅打印信息)。

    返回:
    nltk.probability.FreqDist: 词频分布对象。
    """
    fdist = FreqDist(tokens)
    print(f"最常见的 {num_most_common} 个词语: {fdist.most_common(num_most_common)}")
    
    if save_filename_png:
        plt.figure(figsize=(12, 6))
        fdist.plot(num_most_common, title=plot_title, cumulative=False)
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png)
        plt.savefig(full_save_path)
        print(f"词频分布图已保存至: {full_save_path}")
        plt.close()
    else:
        print("未指定save_filename_png，词频分布图将不会被保存。")
        # 可选：如果需要在非保存时也显示（例如在Jupyter中），取消注释以下行
        # plt.figure(figsize=(12, 6))
        # fdist.plot(num_most_common, title=plot_title, cumulative=False)
        # plt.show()
        # plt.close()
    return fdist

# ==============================================================================
# spaCy 处理函数 (spaCy Processing Functions)
# ==============================================================================

def spacy_process_text(text, nlp_model):
    """
    使用加载的spaCy模型处理文本。

    参数:
    text (str): 要处理的输入文本。
    nlp_model (spacy.lang): 已加载的spaCy语言模型。

    返回:
    spacy.tokens.doc.Doc: 处理后的spaCy Doc对象。如果模型无效则返回None。
    """
    if not nlp_model:
        print("spaCy模型无效，无法处理文本。")
        return None
    return nlp_model(text)

def spacy_get_token_attributes(spacy_doc):
    """
    从spaCy Doc对象中提取每个词元的属性 (文本, 词元, 词性, 标签, 是否为停用词)。

    参数:
    spacy_doc (spacy.tokens.doc.Doc): spaCy Doc对象。

    返回:
    list of dict: 每个词元的属性列表。
    """
    attributes = []
    if not spacy_doc:
        return attributes
    for token in spacy_doc:
        attributes.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'is_stopword': token.is_stop
        })
    return attributes

def spacy_get_named_entities(spacy_doc):
    """
    从spaCy Doc对象中提取命名实体。

    参数:
    spacy_doc (spacy.tokens.doc.Doc): spaCy Doc对象。

    返回:
    list of dict: 每个命名实体的属性列表 (文本, 起始位置, 结束位置, 标签)。
    """
    entities = []
    if not spacy_doc or not spacy_doc.ents:
        return entities
    for ent in spacy_doc.ents:
        entities.append({
            'text': ent.text,
            'start_char': ent.start_char,
            'end_char': ent.end_char,
            'label': ent.label_
        })
    return entities

# ==============================================================================
# 词云生成 (Word Cloud Generation)
# ==============================================================================

def generate_wordcloud(text, stopwords_list=None, language='english', 
                       width=800, height=400, background_color='white',
                       save_filename_png=None, **wordcloud_kwargs):
    """
    根据输入文本生成词云图像。

    参数:
    text (str or list of str): 输入文本。可以是单个字符串或字符串列表 (将被连接)。
    stopwords_list (list of str, optional): 停用词列表。如果None，则尝试使用NLTK的默认英语停用词。
    language (str, optional): 用于NLTK停用词的语言。默认为'english'。
    width (int, optional): 词云图像宽度。默认为800。
    height (int, optional): 词云图像高度。默认为400。
    background_color (str, optional): 词云背景色。默认为'white'。
    save_filename_png (str, optional): 词云图像的保存文件名 (例如 'my_wordcloud.png')。
                                     如果提供，则保存图表；否则尝试显示。
    **wordcloud_kwargs: 传递给 wordcloud.WordCloud() 的其他参数。

    返回:
    wordcloud.WordCloud: 生成的词云对象。如果文本为空则返回None。
    """
    if isinstance(text, list):
        processed_text = " ".join(text)
    else:
        processed_text = text

    if not processed_text.strip():
        print("输入文本为空，无法生成词云。")
        return None

    if stopwords_list is None:
        ensure_nltk_resources(['stopwords'])
        try:
            stopwords_list = set(stopwords.words(language))
        except Exception as e:
            print(f"加载 '{language}' 语种的NLTK停用词失败: {e}。将不使用停用词。")
            stopwords_list = set()
    
    wc = WordCloud(width=width, height=height, 
                   background_color=background_color, 
                   stopwords=stopwords_list,
                   **wordcloud_kwargs).generate(processed_text)
    
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png)
        plt.savefig(full_save_path)
        print(f"词云图像已保存至: {full_save_path}")
    else:
        print("未指定save_filename_png，词云图像将尝试显示 (如果环境支持)。")
        plt.show() # 在脚本中可能不会按预期工作，或阻塞
    plt.close()
    return wc

# ==============================================================================
# Scikit-learn 文本特征提取 (Scikit-learn Text Feature Extraction)
# ==============================================================================

def sklearn_count_vectorize(corpus, language='english', **count_vec_kwargs):
    """
    使用Scikit-learn的CountVectorizer对文本语料库进行向量化。

    参数:
    corpus (list of str): 文档语料库列表。
    language (str, optional): 停用词语言。默认为'english'。
    **count_vec_kwargs: 传递给 CountVectorizer 的其他参数 
                         (例如: max_df, min_df, ngram_range, max_features)。

    返回:
    tuple: (scipy.sparse.csr_matrix, CountVectorizer实例) 
           分别是计数矩阵和拟合的CountVectorizer对象。
    """
    if 'stop_words' not in count_vec_kwargs:
        count_vec_kwargs['stop_words'] = language
    
    vectorizer = CountVectorizer(**count_vec_kwargs)
    count_matrix = vectorizer.fit_transform(corpus)
    print(f"CountVectorizer: 从 {len(corpus)} 个文档中提取了 {count_matrix.shape[1]} 个特征。")
    return count_matrix, vectorizer

def sklearn_tfidf_vectorize(corpus, language='english', **tfidf_vec_kwargs):
    """
    使用Scikit-learn的TfidfVectorizer对文本语料库进行向量化。

    参数:
    corpus (list of str): 文档语料库列表。
    language (str, optional): 停用词语言。默认为'english'。
    **tfidf_vec_kwargs: 传递给 TfidfVectorizer 的其他参数 
                         (例如: max_df, min_df, ngram_range, max_features, use_idf, smooth_idf)。

    返回:
    tuple: (scipy.sparse.csr_matrix, TfidfVectorizer实例) 
           分别是TF-IDF矩阵和拟合的TfidfVectorizer对象。
    """
    if 'stop_words' not in tfidf_vec_kwargs:
        tfidf_vec_kwargs['stop_words'] = language
        
    vectorizer = TfidfVectorizer(**tfidf_vec_kwargs)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"TfidfVectorizer: 从 {len(corpus)} 个文档中提取了 {tfidf_matrix.shape[1]} 个特征。")
    return tfidf_matrix, vectorizer

# ==============================================================================
# 概念性演示函数 (Conceptual Demonstration Functions)
# ==============================================================================

def conceptual_gensim_topic_modeling():
    """
    Gensim主题建模 (如LDA) 的概念性概述。
    """
    print("\n--- Gensim 主题建模 (概念性概述) ---")
    print("Gensim用于高级主题建模 (例如LDA) 和词向量 (Word2Vec, FastText)。")
    print("基本步骤:")
    print("  1. 文本预处理: 分词, 词形还原, 去停用词, 清理。")
    print("  2. 创建Gensim词典 (Dictionary): 将词语映射到ID。")
    print("     `from gensim.corpora import Dictionary`")
    print("     `dictionary = Dictionary(processed_corpus_tokens_list)`")
    print("  3. 创建Gensim语料库 (Corpus): 将词典转换为词袋表示。")
    print("     `corpus_bow = [dictionary.doc2bow(doc_tokens) for doc_tokens in processed_corpus_tokens_list]`")
    print("  4. 训练LDA模型: `from gensim.models import LdaModel`")
    print("     `lda_model = LdaModel(corpus_bow, num_topics=5, id2word=dictionary, passes=10)`")
    print("  5. 查看主题: `lda_model.print_topics()`")
    print("  6. (可选) 使用pyLDAvis进行可视化: `import pyLDAvis.gensim_models as gensimvis`")
    print("     `vis_data = gensimvis.prepare(lda_model, corpus_bow, dictionary)`")
    print("     `pyLDAvis.save_html(vis_data, os.path.join(OUTPUT_DIR, 'lda_visualization.html'))`")

def conceptual_scattertext_visualization():
    """
    Scattertext可视化不同类别语料库中词语使用差异的概念性概述。
    """
    print("\n--- Scattertext 词语使用差异可视化 (概念性概述) ---")
    print("Scattertext用于可视化不同类别文本中词语使用频率和关联性的差异。")
    print("基本步骤:")
    print("  1. 准备数据: Pandas DataFrame，包含文本列和类别列。")
    print("  2. (推荐) 使用spaCy进行文本处理以获得高质量的词形和词性。")
    print("  3. 创建Scattertext语料库: `import scattertext as st`")
    print("     `nlp = spacy.load('en_core_web_sm')` (或其他模型)")
    print("     `corpus = st.CorpusFromPandas(df, category_col='类别列名', text_col='文本列名', nlp=nlp).build()`")
    print("  4. 生成交互式HTML可视化图表: ")
    print("     `html = st.produce_scattertext_explorer(corpus,`")
    print("            `category='类别A', category_name='类别A名称', not_category_name='其他类别名称',`")
    print("            `minimum_term_frequency=5, pmi_threshold_coefficient=3, width_in_pixels=1000)`")
    print("  5. 保存HTML: `with open(os.path.join(OUTPUT_DIR, 'scattertext_visualization.html'), 'w') as f: f.write(html)`")

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================

def run_text_analysis_demos():
    """运行所有文本分析与可视化演示函数。"""
    print(f"--- 文本数据分析与可视化接口化演示 (输出文件将保存到 '{OUTPUT_DIR}' 目录) ---")

    sample_corpus = [
        "This is the first document about social science research methods.",
        "The second document concerns Python programming and advanced data analysis techniques.",
        "Social science and Python can be powerfully combined for various research purposes.",
        "Data analysis in the field of social science often involves statistical methods and tools.",
        "Natural Language Processing (NLP) is an exciting subfield of artificial intelligence and linguistics."
    ]

    # 0. 确保资源和加载模型
    print("\n--- 0. 资源准备 ---")
    ensure_nltk_resources() # 检查默认资源
    nlp_spacy_model = load_spacy_model() # 尝试加载默认的 'en_core_web_sm'

    # 1. NLTK 处理演示
    print("\n--- 1. NLTK 处理演示 ---")
    doc1_tokens = nltk_tokenize_text(sample_corpus[0])
    print(f"  文档1分词结果 (NLTK): {doc1_tokens[:10]}...")
    doc1_pos_tags = nltk_pos_tag_tokens(doc1_tokens)
    print(f"  文档1词性标注 (NLTK): {doc1_pos_tags[:10]}...")
    
    all_corpus_tokens = []
    for doc_text in sample_corpus:
        tokens = nltk_tokenize_text(doc_text, lower=True)
        all_corpus_tokens.extend([word for word in tokens if word.isalpha()]) # 简单过滤
    nltk_freq_dist(all_corpus_tokens, num_most_common=15, plot_title="语料库高频词 (NLTK)", save_filename_png="demo_nltk_freq_dist.png")

    # 2. spaCy 处理演示
    if nlp_spacy_model:
        print("\n--- 2. spaCy 处理演示 ---")
        spacy_doc2 = spacy_process_text(sample_corpus[1], nlp_spacy_model)
        if spacy_doc2:
            token_attrs = spacy_get_token_attributes(spacy_doc2)
            print("  文档2词元属性 (spaCy, 前5个):")
            for attr in token_attrs[:5]: print(f"    {attr}")
            
            entities = spacy_get_named_entities(spacy_doc2)
            print("  文档2命名实体 (spaCy):")
            if entities: 
                for ent in entities: print(f"    {ent}")
            else: print("    未找到命名实体。")
    else:
        print("\n--- 2. spaCy 处理演示 (已跳过，模型未加载) ---")

    # 3. 词云生成演示
    print("\n--- 3. 词云生成演示 ---")
    generate_wordcloud(" ".join(sample_corpus), 
                       save_filename_png="demo_corpus_wordcloud.png",
                       colormap='viridis', min_font_size=10)

    # 4. Scikit-learn 特征提取演示
    print("\n--- 4. Scikit-learn 特征提取演示 ---")
    count_matrix, count_vec = sklearn_count_vectorize(sample_corpus, max_features=15, ngram_range=(1,2))
    if hasattr(count_vec, 'get_feature_names_out'):
        print(f"  CountVectorizer 特征名 (前10): {count_vec.get_feature_names_out()[:10]}")
    print(f"  CountVectorizer 矩阵形状: {count_matrix.shape}")

    tfidf_matrix, tfidf_vec = sklearn_tfidf_vectorize(sample_corpus, max_features=15, use_idf=True)
    if hasattr(tfidf_vec, 'get_feature_names_out'):
        print(f"  TfidfVectorizer 特征名 (前10): {tfidf_vec.get_feature_names_out()[:10]}")
    print(f"  TfidfVectorizer 矩阵形状: {tfidf_matrix.shape}")

    # 5. 概念性演示
    print("\n--- 5. 其他高级库概念性演示 ---")
    conceptual_gensim_topic_modeling()
    conceptual_scattertext_visualization()

    print(f"\n--- 文本分析演示完成。输出文件已保存到 '{OUTPUT_DIR}' 目录。 ---")

if __name__ == '__main__':
    run_text_analysis_demos() 