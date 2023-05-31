import openai
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
import pandas as pd
import os
import scipdf ## You need a Gorbid service available
import tabula ## You need to have the Java Tabula installed in the environment
from gradio import DataFrame
import asyncio
from transformers import pipeline
from dotenv import load_dotenv
import json
load_dotenv()

## You api key from vendors or hugginface
openai.api_key=os.getenv("OPEN_AI_API_KEY")
LLMClient = OpenAI(model_name='text-davinci-003', openai_api_key=openai.api_key,temperature=0)

# Extract text from PDF file using SCIPDF and Gorbid service (you need gorbid to use it)
def extract_text_from_pdf(file_path):
    article_dict = scipdf.parse_pdf_to_dict(file_path, soup=True,return_coordinates=False, grobid_url="https://kermitt2-grobid.hf.space") # return dictionary
    print("parsed")
    source = article_dict.find("sourcedesc")
    authors = source.find_all("persname")
    finaltext = article_dict['title'] + " \n\n " + article_dict['authors'] + " \n\n Abstract: " + article_dict['abstract'] + " \n\n "
    sections = []
    for section in article_dict['sections']:
        sec = section['heading'] + ": "
        if(isinstance(section['text'], str)):
            finaltext = finaltext + sec + section['text'] + " \n\n " 
        else:
            for text in section['text']:
                sec = sec + text+ " \n\n " 
            finaltext = finaltext + sec
    return finaltext

# Extract and transform the tables of the papers
async def get_tables(docsearch,chain_table,input_file):   
    print("Getting tables")
    table_texts = []
    dfs = tabula.read_pdf(input_file.name, pages='all')
    for idx, table in enumerate(dfs):
        query = "Table "+str(idx+1)+":"
        docs = docsearch.similarity_search(query, k=4)
        #result = chain_table({"context":docs,"table":table})
        table_texts.append(async_table_generate(docs, table,  chain_table))
        #print(query + " "+ result['text'])
        #table_texts.append(query + " "+ result['text'])
    table_texts = await asyncio.gather(*table_texts)
    for table in table_texts:
        docsearch.add_texts(table[1])
    return docsearch

def extract_text_clean(file_path):
    file_extension = os.path.splitext(file_path.name)[1]
    if file_extension == ".pdf":
        all_text = extract_text_from_pdf(file_path.name)
        return all_text
    elif file_extension == ".txt":
        with open(file_path.name) as f:
            all_text = f.read()
            return all_text

async def prepare_data(input_file, chain_table, apikey):
    #with open(input_file.name) as f:
    #    documentation = f.read()
    file_name = input_file.name.split("/")[-1]
 

    # Process text and get the embeddings
    filepath = "./vectors/"+file_name
    if not apikey:
        apikey = openai.api_key
        gr.Error("Please set your api key")
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    if os.path.isfile(filepath+"/index.faiss"):

        # file exists
        docsearch = FAISS.load_local(filepath,embeddings=embeddings)

        print("We get the embeddings from local store")
    else:
        #progress(0.40, desc="Detected new document. Splitting and generating the embeddings")
        print("We generate the embeddings using thir-party service")
        # Get extracted running text
        text = extract_text_clean(input_file)

        # Configure the text splitter and embeddings
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=10, separators=[".", ",", " \n\n "])

        # Split, and clean
        texts = text_splitter.split_text(text)
        for idx, text in enumerate(texts):
            texts[idx] = text.replace('\n',' ')
        print("Creating embeddings")
        # Create an index search    
        docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
        
        # Extract and prepare tables
       # progress(0.60, desc="Embeddings generated, parsing and transforming tables")
        if (os.path.splitext(input_file.name)[1] == '.pdf'):
            docsearch = await get_tables(docsearch,chain_table,input_file)
        
        # Save the index locally
        FAISS.save_local(docsearch, "./vectors/"+file_name)
 
    return docsearch

def build_chains(apikey):
    if not apikey:
        apikey = openai.api_key
        gr.Error("Please set your api key")
    LLMClient = OpenAI(model_name='text-davinci-003',openai_api_key=apikey,temperature=0)
    ## In-context prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Question: {question}
    ###
    Context: 
    {context}
    ###
    Helpful answer:
    """
    in_context_prompt = PromptTemplate(
        input_variables=["context","question"],
        template=prompt_template,
    )
    chain_incontext = load_qa_chain(LLMClient, chain_type="stuff", prompt=in_context_prompt)

    # Table extraction prompts
    ## Table prompt to transform parsed tables in natural text
    prompt_template = """Given the following table in HTML, and the given context related the table: Translate the content of the table into natural language.
    ###
    Context: 
    {context}
    ###
    Table: {table}
    ###
    Table translation:
    """
    table_prompt = PromptTemplate(
        input_variables=["context","table"],
        template=prompt_template,
    )
    chain_table = LLMChain(llm=LLMClient, prompt=table_prompt)

    return chain_incontext, chain_table

async def async_table_generate(docs,table,chain):

    resp = await chain.arun({"context": docs, "table": table})
    #resp = "Description of the team, the type, and the demographics information, Description of the team, the type, and the demographics information"
    return resp

async def async_generate(dimension, docs,question,chain):
    resp = await chain.arun({"input_documents": docs, "question": question})
    #resp = "Description of the team, the type, and the demographics information, Description of the team, the type, and the demographics information"
    return [dimension, resp]

async def get_gathering_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Gathering description":"""Provide a summary of how the data of the dataset has been collected? Please avoid mention the annotation process or data preparation processes"""},
                 {"Gathering type":"""Which of the following types corresponds to the gathering process mentioned in the context?

Types: Web API, Web Scrapping, Sensors, Manual Human Curator, Software collection, Surveys, Observations, Interviews, Focus groups, Document analysis, Secondary data analysis, Physical data collection, Self-reporting, Experiments, Direct measurement, Interviews, Document analysis, Secondary data analysis, Physical data collection, Self-reporting, Experiments, Direct measurement, Customer feedback data, Audio or video recordings, Image data, Biometric data, Medical or health data, Financial data, Geographic or spatial data, Time series data, User-generated content data.

Answer with "Others", if you are unsure. Please answer with only the type"""},
                 {"Gathering team": """Who was the team who collect the data?"""},
                 {"Team Type": """The data was collected by an internal team, an external team, or crowdsourcing team?""" },
                 {"Team Demographics": "Are the any demographic information of team gathering the data?"},
                 {"Timeframe ":""" Which are the timeframe when the data was collected?
                    If present, answer only with the collection timeframe of the data. If your are not sure, or there is no mention, just answers 'not provided'"""},
                 {"Sources": """Which is the source of the data during the collection process? Answer solely with the name of the source""" },
                 {"Infrastructure": """Which tools or infrastructure has been used during the collection process?"""},
                 {"Localization": """Which are the places where data has been collected?
                 If present, answer only with the collection timeframe of the data. If your are not sure, or there is no mention, just answers 'not provided'"""}
    
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

async def get_annotation_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Annotation description":"""How the data of the has been annotated or labelled? Provide a short summary of the annotation process"""},
                 {"Annotation type":""" Which  of the following category corresponds to the annotation
                process mentioned in the context? 
                
                Categories: Bounding boxes, Lines and splines, Semantinc Segmentation, 3D cuboids, Polygonal segmentation, Landmark and key-point, Image and video annotations, Entity annotation, Content and textual categorization
                
                If you are not sure, answer with 'others'. Please answer only with the categories provided in the context. """},
                 {"Labels":""" Which are the specific labels of the dataset? Can you enumerate it an provide a description of each one?"""},
                 {"Team Description": """Who has annotated the data?"""},
                 {"Team type": """The data was annotated by an internal team, an external team, or crowdsourcing team?""" },
                 {"Team Demographics": """Is there any demographic information about the team who annotate the data?"""},
                 {"Infrastructure": """Which tool has been used to annotate or label the dataset?"""},
                 {"Validation": """How the quality of the labels have been validated?""" }
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers
  
async def get_social_concerns_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Representetiveness":"""Are there any social group that could be misrepresented in the dataset?"""},
                 {"Biases":"""Is there any potential bias or imbalance in the data?"""},
                 {"Sensitivity":""" Are there sensitive data, or data that can be offensive for people in the dataset?"""},
                 {"Privacy":""" Is there any privacy issues on the data?"""},
                
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

async def get_uses_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Purposes":"""Which are the purpose or purposes of the dataset?"""},
                 {"Gaps":"""Which are the gaps the  dataset intend to fill?"""},
                 {"Task":"""Which machine learning tasks the dataset inteded for?:"""},
                 {"Recommended":"""For which applications the dataset is recommended?"""},
                 {"Non-Recommneded":"""Is there any non-recommneded application for the dataset? If you are not sure, or there is any non-recommended use of the dataset metioned in the context, just answer with "no"."""},
    ]
    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          if (title == "Task"):
            question = """Which of the following ML tasks for the dataset best matches the context?  
            
                    Tasks: text-classification, question-answering, text-generation, token-classification, translation,
                    fill-mask, text-retrieval, conditional-text-generation, sequence-modeling, summarization, other,
                    structure-prediction, information-retrieval, text2text-generation, zero-shot-retrieval,
                    zero-shot-information-retrieval, automatic-speech-recognition, image-classification, speech-processing,
                    text-scoring, audio-classification, conversational, question-generation, image-to-text, data-to-text,
                    classification, object-detection, multiple-choice, text-mining, image-segmentation, dialog-response-generation,
                    named-entity-recognition, sentiment-analysis, machine-translation, tabular-to-text, table-to-text, simplification,
                    sentence-similarity, zero-shot-classification, visual-question-answering, text_classification, time-series-forecasting,
                    computer-vision, feature-extraction, symbolic-regression, topic modeling, one liner summary, email subject, meeting title,
                    text-to-structured, reasoning, paraphrasing, paraphrase, code-generation, tts, image-retrieval, image-captioning,
                    language-modelling, video-captionning, neural-machine-translation, transkation, text-generation-other-common-sense-inference,
                    text-generation-other-discourse-analysis, text-to-tabular, text-generation-other-code-modeling, other-text-search

                    If you are not sure answer with just with "others".
                    Please, answer only with one or some of the provided tasks """
            
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

async def get_contributors_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Authors":"""Who are the authors of the dataset """},
                 {"Funders":"""Is there any organization which supported or funded the creation of the dataset?"""},
                 {"Maintainers":"""Who are the maintainers of the dataset?"""},
                 {"Erratums":"""Is there any data retention limit in the dataset? If you are not sure, or there is no retention limit just answer with "no"."""},
                 {"Data Retention Policies":"""Is there any data retention policies policiy of the dataset? If you are not sure, or there is no retention policy just answer with "no"."""},
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

async def get_composition_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"File composition":"""Can you provide a description of each files the dataset is composed of?"""},
                 {"Attributes":"""Can you enumerate the different attributes present in the dataset? """},
                 {"Trainig splits":"""The paper mentions any recommended data split of the dataset?"""},
                 {"Relevant statistics":"""Are there relevant statistics or distributions of the dataset? """},
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

async def get_distribution_dimension(docsearch, incontext_prompt, retrieved_docs):
    dimensions = [
                 {"Data repository":"""Is there a link to the a repository containing the data? If you are not sure, or there is no link to the repository just answer with "no"."""},
                 {"Licence":"""Which is the license of the dataset. If you are not sure, or there is mention to a license of the dataset in the context, just answer with "no". """},
                 {"Deprecation policies":"""Is there any deprecation plan or policy of the dataset?
                """},
                
    ]

    results = []
    for dimension in dimensions:
        for title, question in dimension.items():
          docs = docsearch.similarity_search(question, k=retrieved_docs)
          results.append(async_generate(title, docs,question,incontext_prompt))

    answers = await asyncio.gather(*results)
    return answers

def get_warnings(results):
    warnings = []
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    for result in results:  
        if(result[0] == "Team Demographics"):
            classifications = classifier(result[1], ["Have demographics information","Do not have demographics information"])
            if(classifications['labels'][0] == 'Do not have demographics information'):
                print("Dimension: "+result[0]+" is missing. Inserting a warning")
                warnings.append(result[0]+" is missing. This information is relevant to evaluate the quality of the labels")
        if(result[0] == "Localization"):  
            classifications = classifier(result[1], ["Is a localization","Is not a localization"])
            if(classifications['labels'][0] == 'Is not a localization'):
                print("Dimension: "+result[0]+" is missing. Inserting a warning")
                warnings.append(result[0]+" is missing. Please indicate where the data has been collected")
        if(result[0] == "Time Localization"):  
            classifications = classifier(result[1], ["It is a date","It is not a date"])
            if(classifications['labels'][0] == 'Is not a localization'):
                print("Dimension: "+result[0]+" is missing. Inserting a warning")
                warnings.append(result[0]+" is missing. Please indicate when the data has been collected")
    if len(warnings) == 0:
        warnings.append("No warnings")
    return warnings

# Define function to handle the Gradio interface 
async def annotate_only(input_file, apikey):
    #progress(0, desc="Starting")
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    #progress(0.20, desc="Preparing data: Generating embeddings, splitting text, and adding transformed tables")
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    #progress(0.40, desc="Extracting dimensions")
    results = await get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    #progress(0.80, desc="Generating Warning")
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)
# Define function to handle the Gradio interface
async def uses_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)

# Define function to handle the Gradio interface
async def distribution_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)
# Define function to handle the Gradio interface
async def social_cocerns_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)
# Define function to handle the Gradio interface
async def composition_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)
# Define function to handle the Gradio interface
async def contributors_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)

    # Build results in the correct format for the Gradio front-end
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return results, gr.update(value=pd.DataFrame(warnings,columns=['Warnings:']), visible=True)

async def gather_only(input_file, apikey):
    # Build the chains
    chain_incontext, chain_table = build_chains(apikey) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey)
    # Get annotation dimensions
    results = await get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get warning
    warnings = get_warnings(results)
    results = pd.DataFrame(results, columns=['Dimension', 'Results'])
    return  results, gr.update(value=pd.DataFrame(warnings, columns=['Warnings:']), visible=True)

async def complete(input_file):

    # Build the chains
    chain_incontext, chain_table = build_chains(apikey=os.getenv("OPEN_AI_API_KEY")) 
    # Prepare the data
    docsearch = await prepare_data(input_file, chain_table, apikey=os.getenv("OPEN_AI_API_KEY"))
    #Retrieve dimensions    
    results = await asyncio.gather(get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10))
    # Get warning from the results
    warnings = []
    for result in results:
        warnings.append(gr.update(value=[get_warnings(result)], visible=True))
    #warnings_dt = gr.update(value=pd.DataFrame(warnings,columns=['Warnings:'],labels= None), visible=True)
    results.extend(warnings)
    return results

async def annotation_api_wrapper(input_file, apikey):

    results, alarms = await annotate_only(input_file, apikey)

    response_results = results.to_dict()
    response_alarms = alarms['value'].to_dict()
    api_answer = {'results':response_results, 'warnings': response_alarms}
    return api_answer

## Building the layout of the app
css = """.table-wrap.scroll-hide.svelte-8hrj8a.no-wrap {
    white-space: normal;
}
#component-7 .wrap.svelte-xwlu1w {
    min-height: var(--size-40);
}
div#component-2 h2 {
    color: var(--block-label-text-color);
    text-align: center;
    border-bottom: 2px solid;
    border-radius: 7px;
    text-align: center;
    margin: 0 15% 0 15%;
}
div#component-5 {
    border: 1px solid var(--border-color-primary);
    border-radius: 0 0px 10px 10px;
    padding: 20px;
}
.gradio-container.gradio-container-3-26-0.svelte-ac4rv4.app {
    max-width: 850px;
}
div#component-6 {
    min-height: 150px;
}
button#component-17 {
    color: var(--block-label-text-color);
}
.gradio-container.gradio-container-3-26-0.svelte-ac4rv4.app {
    max-width: 1100px;
}
#component-9 .wrap.svelte-xwlu1w {
    min-height: var(--size-40);
}
div#component-11 {
    height: var(--size-40);
}
div#component-9 {
    border: 1px solid grey;
    border-radius: 10px;
    padding: 3px;
    text-align: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
            gr.Markdown("## Dataset documentation analyzer")
    with gr.Row():
        gr.Markdown("""Extract, in a structured manner, the **[general guidelines](https://knowingmachines.org/reading-list#dataset_documentation_practices)** from the ML community about dataset documentation practices from its scientific documentation. Study and analyze scientific data published in peer-review journals such as: **[Nature's Scientific Data](https://duckduckgo.com)** and **[Data-in-Brief](https://duckduckgo.com)**. Here you have a **[complete list](https://zenodo.org/record/7082126#.ZDaf-OxBz0p)** of data journals suitable to be analyzed with this tool.
                 """)
        
    with gr.Row():
        
        with gr.Column():
           fileinput = gr.File(label="Upload TXT file"),

        with gr.Column():
             gr.Markdown(""" <h4 style=text-align:center>Instructions: </h4> 
     
         <b>  &#10549; Try the examples </b> at the bottom 

         <b> &#8680; Set your API KEY </b> of OpenAI  
        
         <b> &#8678; Upload </b> your data paper (in PDF or TXT)

         <b> &#8681; Click in get insights  </b> in one tab!

         """)
        with gr.Column():
            apikey_elem = gr.Text(label="Your OpenAI APIKey")
         #   gr.Markdown(""" 
         #                   <h3> Improving your data and assesing your dataset documentation </h3>
         #                   The generated warning also allows you quicly check the completeness of the documentation, and spotting gaps in the document
         #                   <h3> Performing studies studies over scientific data </h3>
         #                   If you need to analyze a large scale of documents, we provide an <strong>API</strong> that can be used programatically. Documentation on how to use it is at the bottom of the page.  """)
    with gr.Row():
        with gr.Tab("Annotation"):
       
            gr.Markdown("""In this chapter, we get information regarding the annotation process of the data: We provide a description of the process and we infer its type from the documentation. Then we extract the labels, and information about the annotation team, the infrastructure used to annotate the data and the validation process applied over the labels""")
            result_anot = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_anot = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_annotation = gr.Button("Get the annotation process insights!")
                
        with gr.Tab("Gathering"):
            gr.Markdown("""In this chapter, we get information regarding the collection process of the data: We provide a description of the process and we infer its type from the documentation. Then we extract information about the collection team, the infrastructure used to collect the data and the sources. Also we get the timeframe of the data collection and its geolocalization.""")
            result_gather = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_gather = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_gathering = gr.Button("Get the gathering process insights!")
        with gr.Tab("Uses"):
            gr.Markdown("""In this chapter, we extract the design intentios of the authors, we extract the purposes, gaps, and we infer the ML tasks (extracted form hugginface) the dataset is inteded for. Also we get the uses recomendation and the ML Benchmarks if the dataset have been tested with them""")
            result_uses = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_uses = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_uses = gr.Button("Get the uses of the dataset!")
        with gr.Tab("Contributors"):
            gr.Markdown("""In this chapter, we extract all the contributors, funding information and maintenance of the dataset""")
            result_contrib = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_contrib = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_contrib = gr.Button("Get the contributors of the dataset!")
          
        with gr.Tab("Composition"):
            gr.Markdown("""In this chapter, we extract the file structure, we identify the attributes of the dataset, the recommneded trainig splits and the relevant statistics (if provided in the documentation) """)
            result_comp = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_comp = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_comp = gr.Button("Get the composition of the dataset!")
        with gr.Tab("Social Concerns"):
            gr.Markdown("""In this chapter, we extract social concerns regarding the representativeness of social groups, potential biases, sensitivity issues, and privacy issues. """)
            result_social = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_social = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_social = gr.Button("Get the Social Cocerns!")

        with gr.Tab("Distribution"):
            gr.Markdown("""In this chapter, we aim to extract the legal conditions under the dataset is released) """)
            result_distri = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_distribution = gr.DataFrame(headers=["warning"],type="array", visible=False)
            button_dist = gr.Button("Get the Distribution!")
    with gr.Row():
        examples = gr.Examples(
                examples=["sources/Nature-Scientific-Data/A whole-body FDG-PET:CT.pdf","sources/Nature-Scientific-Data/Lontar-Manuscripts.pdf"],
                inputs=[fileinput[0]], 
                fn=complete, 
                outputs=[
                    result_anot,
                    result_gather,
                    result_uses,
                    result_contrib,
                    result_comp,
                    result_social,
                    result_distri,
                    alerts_anot,
                    alerts_gather,
                    alerts_uses,
                    alerts_contrib,
                    alerts_comp,
                    alerts_social,
                    alerts_distribution], 
                cache_examples=True)
        button_complete = gr.Button("Get all the dimensions", visible=False)
    allres = gr.Text(visible=False)
    ## Events of the apps
    button_annotation.click(annotate_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_anot,alerts_anot])
    button_gathering.click(gather_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_gather,alerts_gather])
    button_uses.click(uses_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_uses,alerts_uses])
    button_contrib.click(contributors_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_contrib,alerts_contrib])
    button_comp.click(composition_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_comp,alerts_comp])
    button_social.click(social_cocerns_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_social,alerts_social])
    button_dist.click(distribution_only,inputs=[fileinput[0],apikey_elem ],outputs=[result_distri,alerts_distribution])
   

    ## API endpoints
    button_complete.click(annotation_api_wrapper,inputs=[fileinput[0],apikey_elem],outputs=allres, api_name="our")
    
   
    # Run the app
    #demo.queue(concurrency_count=5,max_size=20).launch()
    demo.launch(share=True,auth=("CKIM2023", "demodemo"))
        
