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
from extractor import Extractor
load_dotenv()

## You api key from vendors or hugginface
openai.api_key=os.getenv("OPEN_AI_API_KEY")
LLMClient = OpenAI(model_name='text-davinci-003', openai_api_key=openai.api_key,temperature=0)
extractor = Extractor()

# Define function to handle the Gradio interface 
async def extraction(input_file, apikey, dimension):
    # Build the chains
    chain_incontext, chain_table = extractor.build_chains(apikey) 
    # Prepare the data
    docsearch = await extractor.prepare_data(input_file, chain_table, apikey)
    # Extract dimensions
    if (dimension == "annotation"): 
        results, completeness_report = await extractor.get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "gathering"):
        results, completeness_report = await extractor.get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "uses"):
        results, completeness_report = await extractor.get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "contrib"):
        results, completeness_report = await extractor.get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "comp"):
        results, completeness_report = await extractor.get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "social"):
        results, completeness_report = await extractor.get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10)
    elif (dimension == "dist"):
        results, completeness_report = await extractor.get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10)
    # Get completeness report
    #completeness_report = extractor.postprocessing(results)
    return results, completeness_report

async def ui_extraction(input_file, apikey, dimension):
        results, completeness_report = await extraction(input_file, apikey, dimension)
        # Build results in the correct format for the Gradio front-end
        results = pd.DataFrame(results, columns=['Dimension', 'Results'])
        return results, gr.update(value=pd.DataFrame(completeness_report['report'],columns=['Completeness report: '+str(completeness_report['completeness'])+'%']), visible=True)

async def api_extraction(input_file, apikey, dimension):
    results, completeness_report = extraction(input_file, apikey, dimension)
    response_results = results.to_dict()
    response_alarms = completeness_report['value'].to_dict()
    api_answer = {'results':response_results, 'warnings': response_alarms}
    return api_answer

async def complete(input_file):
    # Build the chains
    chain_incontext, chain_table = extractor.build_chains(apikey=os.getenv("OPEN_AI_API_KEY")) 
    # Prepare the data
    docsearch = await extractor.prepare_data(input_file, chain_table, apikey=os.getenv("OPEN_AI_API_KEY"))
    #Retrieve dimensions    
    results = await asyncio.gather(extractor.get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    extractor.get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10))
    # Get completeness report from the results
    warnings = []
    for result in results:
        warnings.append(gr.update(value=[extractor.postprocessing(result)], visible=True))
    results.extend(warnings)
    return results

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
            gr.Markdown("## DataDoc Analyzer")
    with gr.Row():
        gr.Markdown("""Extract, in a structured manner, the **[general guidelines](https://knowingmachines.org/reading-list#dataset_documentation_practices)** from the ML community about dataset documentation practices from its scientific documentation. Study and analyze scientific data published in peer-review journals such as: **[Nature's Scientific Data](https://www.nature.com/sdata/)** and **[Data-in-Brief](https://www.data-in-brief.com)**. Here you have a **[complete list](https://zenodo.org/record/7082126#.ZDaf-OxBz0p)** of data journals suitable to be analyzed with this tool.
                 """)
        
    with gr.Row():
        
        with gr.Column():
           fileinput = gr.File(label="Upload the dataset documentation"),

        with gr.Column():
             gr.Markdown(""" <h4 style=text-align:center>Instructions: </h4> 
     
         <b>  &#10549; Try the examples </b> at the bottom 

         <b> &#8680; Set your API key </b> of OpenAI  
        
         <b> &#8678; Upload </b> your data paper (in PDF or TXT)

         <b> &#8681; Click in get insights  </b> in one tab!

         """)
        with gr.Column():
            apikey_elem = gr.Text(label="Your OpenAI API key")
         #   gr.Markdown(""" 
         #                   <h3> Improving your data and assesing your dataset documentation </h3>
         #                   The generated warning also allows you quicly check the completeness of the documentation, and spotting gaps in the document
         #                   <h3> Performing studies studies over scientific data </h3>
         #                   If you need to analyze a large scale of documents, we provide an <strong>API</strong> that can be used programatically. Documentation on how to use it is at the bottom of the page.  """)
    with gr.Row():
        with gr.Tab("Annotation"):
       
            gr.Markdown("""In this dimension, you can get information regarding the annotation process of the data: Extract a description of the process and infer its type. Extract the labels and information about the annotation team, the infrastructure used to annotate the data, and the validation process applied to the labels.""")
            result_anot = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_anot = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_annotation = gr.Button("Get the annotation process insights!")
                
        with gr.Tab("Gathering"):
            gr.Markdown("""In this dimension, we get information regarding the collection process of the data: We provide a description of the process and we infer its type from the documentation. Then we extract information about the collection team, the infrastructure used to collect the data and the sources. Also we get the timeframe of the data collection and its geolocalization.""")
            result_gather = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_gather = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_gathering = gr.Button("Get the gathering process insights!")
        with gr.Tab("Uses"):
            gr.Markdown("""In this dimension, we extract the design intentios of the authors, we extract the purposes, gaps, and we infer the ML tasks (extracted form hugginface) the dataset is inteded for. Also we get the uses recomendation and the ML Benchmarks if the dataset have been tested with them""")
            result_uses = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_uses = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_uses = gr.Button("Get the uses of the dataset!")
        with gr.Tab("Contributors"):
            gr.Markdown("""In this dimension, we extract all the contributors, funding information and maintenance of the dataset""")
            result_contrib = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_contrib = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_contrib = gr.Button("Get the contributors of the dataset!")
          
        with gr.Tab("Composition"):
            gr.Markdown("""In this dimension, we extract the file structure, we identify the attributes of the dataset, the recommneded trainig splits and the relevant statistics (if provided in the documentation) """)
            result_comp = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_comp = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_comp = gr.Button("Get the composition of the dataset!")
        with gr.Tab("Social Concerns"):
            gr.Markdown("""In this dimension, we extract social concerns regarding the representativeness of social groups, potential biases, sensitivity issues, and privacy issues. """)
            result_social = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            alerts_social = gr.DataFrame(headers=["warnings"],type="array", visible=False)
            button_social = gr.Button("Get the Social Cocerns!")

        with gr.Tab("Distribution"):
            gr.Markdown("""In this dimension, we aim to extract the legal conditions under the dataset is released) """)
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
    button_annotation.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="annotation")],outputs=[result_anot,alerts_anot])
    button_gathering.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("gathering") ],outputs=[result_gather,alerts_gather])
    button_uses.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("uses") ],outputs=[result_uses,alerts_uses])
    button_contrib.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("contrib") ],outputs=[result_contrib,alerts_contrib])
    button_comp.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("comp") ],outputs=[result_comp,alerts_comp])
    button_social.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("social") ],outputs=[result_social,alerts_social])
    button_dist.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State("dist") ],outputs=[result_distri,alerts_distribution])
   

    ## API endpoints
    api_annotation = gr.Button(visible=False)
    api_annotation.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="annotation")],outputs=[result_anot,alerts_anot], api_name="annotation")
    api_gathering = gr.Button(visible=False)
    api_gathering.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="gathering")],outputs=[result_anot,alerts_anot], api_name="gathering")
    api_uses = gr.Button(visible=False)
    api_uses.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="uses")],outputs=[result_anot,alerts_anot], api_name="uses")
    api_contrib = gr.Button(visible=False)
    api_contrib.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="contrib")],outputs=[result_anot,alerts_anot], api_name="contrib")
    api_comp = gr.Button(visible=False)
    api_comp.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="comp")],outputs=[result_anot,alerts_anot], api_name="composition")
    api_social = gr.Button(visible=False)
    api_social.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="social")],outputs=[result_anot,alerts_anot], api_name="social")
    api_dist = gr.Button(visible=False)
    api_dist.click(api_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="dist")],outputs=[result_anot,alerts_anot], api_name="dist")

    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    #button_complete.click(api_extraction,inputs=[fileinput[0],apikey_elem,"annotation"],outputs=allres, api_name="annotation")
    
   
    # Run the app
    #demo.queue(concurrency_count=5,max_size=20).launch()
    demo.launch(share=False,auth=("CIKM2023", "demodemo"))
        
