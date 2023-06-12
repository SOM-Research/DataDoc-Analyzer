# DataDoc Analyzer



Extract, in a structured manner, the **[general guidelines](https://knowingmachines.org/reading-list#dataset_documentation_practices)** from the ML community about dataset documentation practices from its scientific documentation. Study and analyze scientific data published in peer-review journals such as: **[Nature's Scientific Data](https://www.nature.com/sdata/)** and **[Data-in-Brief](https://www.data-in-brief.com)**. 

Here you have a **[complete list](https://zenodo.org/record/7082126#.ZDaf-OxBz0p)** of data journals suitable to be analyzed with this tool. Test the web UI of the tool in the following **[HuggingFace Space](https://huggingface.co/spaces/JoanGiner/DataDoc_Analyzer)**


## ⚒️ Installation

The tools comes with two UI's. A web app built with gradio inteded to test the capabilities of the tool and to analyze a single document (you can test it in the HuggingFace Space). And, a API built with FastAPI, suited to be integrated in any ML pipeline:

To use this tool you need to have **python3.10**, **git**, and **pip** installed in your system.

Then just:
```
git clone https://github.com/JoanGi/DataDoc-Analyzer.git datadoc

## Enter to the created folder
cd datadoc

## Install dependencies
## Better to do this in a virtual enviroment
pip install -r requirements.txt
```
To deploy the web UI:
```
python3 app.py
```
To deploy the API:
```
uvicorn api:app 
```



## ☑️ Usage

### Web UI

To use this tool you need to provide your own API key form OpenAI. 

Once setted, you can upload your PDF from one of the scientífic journals suited for this tool[^1]. Keep en mind, that we analyze "data papers", other publication's type present in these journals, such as "meta-analysis", will not work properly.

At last, click in "get insights" of any tab and you will get the results together with the completeness report.


[^1]: Some journals that publish data papers:
 **[Nature's Scientific Data](https://www.nature.com/sdata/)**, **[Data-in-Brief](https://www.data-in-brief.com)**, **[Geoscience Data Journal](https://rmets.onlinelibrary.wiley.com/journal/20496060)** etc... Here you have a **[complete list](https://zenodo.org/record/7082126#.ZDaf-OxBz0p)** of data journals suitable to be analyzed with this tool

 <div align="center" style="width:100%">

![Api showcase](./assets/appshort.gif)



</div>

 ### API

 The API imitates the behaivour of the tabs of the web UI, but you also have an endpoints to retrieve all the dimensions at the same time. The swagger documentation of the API, that can be tried in-situ, is published together with the app. The server will start at port 8000 by default (if not occupied by other app of your system). And the documentation will be found at http://127.0.0.1:8000/docs


<div align="center" style="width:100%">

![Api showcase](./assets/apigif.gif)



</div>

## 📚 Background research

This tool is currently under a review process.



## ⚖️ License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>

The [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>


