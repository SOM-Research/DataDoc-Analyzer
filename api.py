# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.extractor import Extractor
import json
import tempfile
import os

app = FastAPI()
extractor = Extractor()

@app.get("/")
async def root():
    return {"message": "API working! go to /Docs to see the Documentation"}

@app.post("/annotation")
async def annotation(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "annotation")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/gathering")
async def gathering(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "gathering")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/uses")
async def uses(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "uses")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/contrib")
async def contrib(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "contrib")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/composition")
async def composition(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "comp")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/social")
async def social(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "social")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/distribution")
async def distribution(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results, completeness_report = await extractor.extraction(file.filename, temp.name, apikey, "dist")
        return JSONResponse([{"Results:":results},{"Completeness Report":completeness_report}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file

@app.post("/complete")
async def complete(file:UploadFile,apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results = await extractor.complete_extraction(file.filename, temp.name, apikey)
        return JSONResponse([{"results:":results}])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file