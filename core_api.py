import uvicorn
import shutil
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database.database_connection import SQLiteDatabase
from data_processor import DataImporter 
from text_classifier import TextClassifier
from core_classifier import CoreClassifierTrain

app = FastAPI()

data_importer = DataImporter()
text_classifier = TextClassifier()
core_classifier_trainer = CoreClassifierTrain()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

class DataClass(BaseModel):
    data_id: str
    label: str

class LabelInput(BaseModel):
    dataset_id: str
    changed_data : List[DataClass]

class FileLocationInput(BaseModel):
    file_location: str

class DataInput(BaseModel):
    dataset_id: str
    data: str
    class_name: str

class MultipleDataLabelInput(BaseModel):
    data: str
    class_name: str

class MultiDataInput(BaseModel):
    dataset_id: str
    data_inputs: List[MultipleDataLabelInput]

class ModelInfo(BaseModel):
    datamodel_id: str
    accuracy: float
    f1_score: float
    dataset_id: str  

class TrainAndEvaluateInput(BaseModel):
    model_name: str
    class_name_list: List[str]
    selected_models: Optional[List[str]] = ['bert', 'albert', 'xlnet']

class DatasetInfo(BaseModel):
    dataset_id: str
    dataset_name: str

class NewClass(BaseModel):
    class_name: str
    dataset: List[str]

class NewData(BaseModel):
    class_name: str
    data: List[str]

class ClassNames(BaseModel):
    class_name_list:List[str]

class ClassifyTextData(BaseModel):
    datamodel_id: str
    text: str

# Endpoint to retrieve information about datasets : TESTED
@app.get("/datasets/info/")
def get_datasets_info():
    try:
        datasets = []

        # Execute query to fetch dataset info
        query = "SELECT dataset_id, dataset_name FROM dataset_info"
        results = SQLiteDatabase().execute_query(query)

        # Process query results
        for row in results:
            dataset = DatasetInfo(dataset_id=row[0], dataset_name=row[1])
            datasets.append(dataset)

        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve information about a single dataset : TESTED
@app.get("/datasets/{dataset_id}/")
def get_single_dataset(dataset_id: str):
    try:
        query = f"SELECT data_id, data, class_name FROM data_info WHERE dataset_id = '{dataset_id}'"
        result = SQLiteDatabase().execute_query(query)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_info = [{"data_id": row[0], "data": row[1], "class_name": row[2]} for row in result]
        return {"dataset_id": dataset_id, "data": dataset_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to add a single class_name to data : TESTED
@app.post("/data/class_name/")
def add_label(label_input: LabelInput):
    try:
        for entry in label_input.changed_data:
            class_name = str(entry.label).upper()
            data_id = str(entry.data_id)

            query = f"UPDATE data_info SET class_name = '{class_name}' WHERE dataset_id = '{label_input.dataset_id}' AND data_id = '{data_id}'"
            SQLiteDatabase().execute_query(query)

            query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{class_name}')"""
            SQLiteDatabase().execute_query(query)

            query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{class_name}','{label_input.dataset_id}')"""
            SQLiteDatabase().execute_query(query)

        return {"message": "class_name added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get all class_name : TESTED
@app.get("/class_names/")
def get_class():
    try:
        result = SQLiteDatabase().get_class_info_table_data()
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get all class_info : TESTED
@app.get("/class_info/")
def get_class_info():
    try:
        query = """
                    SELECT ci.class_name, di.dataset_id, di.dataset_name
                    FROM class_info ci
                    JOIN class_dataset_info cdi ON ci.class_name = cdi.class_name
                    JOIN dataset_info di ON cdi.dataset_id = di.dataset_id
                    ORDER BY ci.class_name
                """
        rows = SQLiteDatabase().execute_query(query)
        class_data = {}
        for row in rows:
            class_name = row[0]
            dataset_id = row[1]
            dataset_name = row[2]
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append({'dataset_id':dataset_id, 'dataset_name':dataset_name})

        result = [{'class_name': class_name, 'datasets': class_data[class_name]} for class_name in class_data]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve information about a single dataset and the classes that it can have : TESTED
@app.get("/datasets/class/{dataset_id}/")
def get_single_dataset(dataset_id: str):
    try:
        query = f"SELECT data_id, data, class_name FROM data_info WHERE dataset_id = '{dataset_id}'"
        result = SQLiteDatabase().execute_query(query)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_info = [{"data_id": row[0], "data": row[1], "class_name": row[2]} for row in result]

        query = f"""SELECT class_name FROM class_dataset_info WHERE dataset_id = '{dataset_id}'"""
        result = SQLiteDatabase().execute_query(query)

        class_names = []
        for class_name in result:
            class_names.append(class_name[0])

        return {"labels": class_names, "data": dataset_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new class : TESTED
@app.post("/class/")
def add_class_name(new_class:NewClass):
    try:
        class_name = str(new_class.class_name).upper()
        datasets = new_class.dataset

        query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{class_name}')"""
        SQLiteDatabase().execute_query(query)

        for dataset_id in datasets:
            query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{class_name}','{dataset_id}')"""
            SQLiteDatabase().execute_query(query)
        return {"message": "Class Created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get classes belong to given class : TESTED
@app.get("/dataset/class")
def get_classes_for_dataset(dataset_id:str):
    try:
        query = f"""SELECT class_name FROM class_dataset_info WHERE dataset_id = '{dataset_id}'"""
        result = SQLiteDatabase().execute_query(query)

        class_names = []
        for class_name in result:
            class_names.append(class_name[0])
        return {"class_names": class_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get datasets belong to given class : TESTED
@app.get("/class/datasets")
def get_classes_for_dataset(class_name:str):
    try:
        query = f"""SELECT dataset_id FROM class_dataset_info WHERE class_name = '{class_name.upper()}'"""
        result = SQLiteDatabase().execute_query(query)

        dataset_ids = []
        for dataset_id in result:
            dataset_ids.append(dataset_id[0])
        return {"dataset_ids": dataset_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/class/dataset")
def update_class_datasets(class_update:NewClass):
    try:
        class_name = str(class_update.class_name).upper()
        datasets = class_update.dataset

        for dataset_id in datasets:
            query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{class_name}','{dataset_id}')"""
            SQLiteDatabase().execute_query(query)

        query = f"""SELECT dataset_id FROM class_dataset_info WHERE class_name = '{class_name.upper()}'"""
        result = SQLiteDatabase().execute_query(query)

        for dataset_id in result:
            if dataset_id not in datasets:
                query = f"""DELETE FROM class_dataset_info WHERE dataset_id = '{dataset_id}' AND class_name = '{class_name.upper()}'"""
            
        return {"message": "Class Created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to import data from file : TESTED
@app.post("/datasets/import/")
def import_data(dataset_name: str = Form(...), file: UploadFile = File(...)):
    try:
        UPLOAD_DIRECTORY = "uploaded_files"
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        # Save the file to the specified location
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = data_importer.import_data_from_file(dataset_name, file_location)

        return {"message": "Data imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to import data from file to given class : 
@app.post("/class/datasets/import/")
def import_data(dataset_name: str = Form(...), class_name: str = Form(...), file: UploadFile = File(...)):
    try:
        UPLOAD_DIRECTORY = "uploaded_files"
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        # Save the file to the specified location
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = data_importer.import_data_from_file(dataset_name, file_location, class_name)

        return {"message": "Data imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to add new data to the dataset
@app.post("/datasets/add_data/")
def add_data(data_input: DataInput):
    try:
        dataset_exists = SQLiteDatabase().execute_query(f"SELECT COUNT(*) FROM dataset_info WHERE dataset_id = '{data_input.dataset_id}'")
        if dataset_exists[0][0] == 0:
            raise HTTPException(status_code=404, detail=f"Dataset with dataset_id {data_input.dataset_id} does not exist.")

        last_data_id_query = f"SELECT MAX(CAST(data_id AS INTEGER)) FROM data_info WHERE dataset_id = '{data_input.dataset_id}'"
        last_data_id_result = SQLiteDatabase().execute_query(last_data_id_query)
        last_data_id = last_data_id_result[0][0] if last_data_id_result[0][0] is not None else 0

        new_data_id = str(int(last_data_id) + 1)

        class_name = str(data_input.class_name).upper()

        new_data_record = {
            'dataset_id': data_input.dataset_id,
            'data_id': new_data_id,
            'data': data_input.data,
            'class_name': class_name
        }
        SQLiteDatabase().insert_record('data_info', new_data_record)

        query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{class_name}')"""
        SQLiteDatabase().execute_query(query)

        query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{class_name}','{data_input.dataset_id}')"""
        SQLiteDatabase().execute_query(query)

        return {"message": f"New data added to dataset {data_input.dataset_id} with data_id {new_data_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to retrieve data for given class
@app.post("/class/data/")
def class_data(class_names : ClassNames):
    
    dataset = {}
    for class_name in class_names.class_name_list:
        query = f"""SELECT data, class_name FROM data_info WHERE class_name = '{class_name.upper()}'"""
        result = SQLiteDatabase().execute_query(query)
        for row in result:
            data, class_name = row
            if class_name in dataset:
                dataset[class_name].append(data)
            else:
                dataset[class_name] = [data]
    return dataset

# API endpoint to retrieve data for given class : TESTED
@app.get("/class/data/{class_name}")
def class_data(class_name:str):

    data_list = []
    query = f"""SELECT data_id, data FROM data_info WHERE class_name = '{class_name.upper()}'"""
    result = SQLiteDatabase().execute_query(query)
    for row in result:
        data_id, data = row
        data_list.append({'data_id':data_id, 'data':data})
    return {'data':data_list}

#API endpoint to delete given class :
@app.delete("/class/{class_name}")
def delete_class(class_name: str):
    try:
        SQLiteDatabase().execute_query(f"DELETE FROM class_info WHERE class_name = '{class_name}'")
        SQLiteDatabase().execute_query(f"DELETE FROM class_dataset_info WHERE class_name = '{class_name}'")
        SQLiteDatabase().execute_query(f"DELETE FROM data_info WHERE class_name = '{class_name}'")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        return False

#API endpoint to add data to given class : TESTED
@app.post("/class/data/add")
def class_data(new_data:NewData):
    class_name = str(new_data.class_name).upper()
    data_list = new_data.data
    for data in data_list:
        dataset_id = '0'
        last_data_id_query = f"SELECT MAX(CAST(data_id AS INTEGER)) FROM data_info WHERE dataset_id = '{dataset_id}'"
        last_data_id_result = SQLiteDatabase().execute_query(last_data_id_query)
        last_data_id = last_data_id_result[0][0] if last_data_id_result[0][0] is not None else 0

        new_data_id = str(int(last_data_id) + 1)

        new_data_record = {
            'dataset_id': dataset_id,
            'data_id': new_data_id,
            'data': data,
            'class_name': class_name
        }
        SQLiteDatabase().insert_record('data_info', new_data_record)

        query = f"""INSERT OR IGNORE INTO class_info (class_name) VALUES ('{class_name}')"""
        SQLiteDatabase().execute_query(query)

        query = f"""INSERT OR IGNORE INTO class_dataset_info (class_name, dataset_id) VALUES ('{class_name}','{dataset_id}')"""
        SQLiteDatabase().execute_query(query)

    return True

# API endpoint to retrieve information about all data inside model_info table
@app.get("/model_info/")
def get_model_info():
    try:
        query = "SELECT datamodel_id, datamodel_name, accuracy, f1_score FROM model_info"
        model_info_records = SQLiteDatabase().execute_query(query)

        query = "SELECT * FROM model_class_info"
        model_class_info_records = SQLiteDatabase().execute_query(query)

        print(f"model_info_records : \n {model_info_records}")
        print(f"model_class_info_records : \n {model_class_info_records}")

        model_info_dict = {}
        for model in model_info_records:
            model_info_dict[model[0]] = {'model_id':model[0], 'model_name':model[1], 'model_accuracy':model[2], 'model_f1':model[3], "scores":[]}

        for model_class in model_class_info_records:
            model_info_dict[model_class[0]]["scores"].append({"class_name":model_class[1], "class_id":model_class[2], "precision":model_class[3], "f1_score":model_class[5]})
        return list(model_info_dict.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        return []

# API endpoint to retrieve information about all data inside model_info table
@app.get("/model_names/")
def get_model_names():
    try:
        query = "SELECT datamodel_id, datamodel_name FROM model_info"
        model_info_records = SQLiteDatabase().execute_query(query)

        print(f"model_info_records : \n {model_info_records}")

        model_info_dict = []
        for model in model_info_records:
            model_info_dict.append({'model_id':model[0], 'model_name':model[1]})

        return model_info_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        return []

# API endpoint to delete a specific row in the model_info table and corresponding model folder
@app.delete("/model_info/{datamodel_id}")
def delete_model_info(datamodel_id: str):
    try:
        model_folder_path = f"models/{datamodel_id}"

        model_exists = SQLiteDatabase().execute_query(f"SELECT COUNT(*) FROM model_info WHERE datamodel_id = '{datamodel_id}'")
        if model_exists[0][0] == 0:
            if os.path.exists(model_folder_path):
                shutil.rmtree(model_folder_path)
                raise HTTPException(status_code=404, detail=f"Model with datamodel_id {datamodel_id} does not exist. But model files were deleted.")
            else:
                raise HTTPException(status_code=404, detail=f"Model with datamodel_id {datamodel_id} does not exist on both database and model folder.")
        else:
            SQLiteDatabase().execute_query(f"DELETE FROM model_info WHERE datamodel_id = '{datamodel_id}'")

        if os.path.exists(model_folder_path):
            shutil.rmtree(model_folder_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model with datamodel_id {datamodel_id} does not exist on folder. But model files were deleted from database.")

        return {"message": f"Model with datamodel_id {datamodel_id} and its corresponding folder deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_and_evaluate/")
def train_and_evaluate(data: TrainAndEvaluateInput):
    try:
        success, message = core_classifier_trainer.train_and_evaluate(data.model_name, data.class_name_list, data.selected_models)
        if success is True:
            return {"Success":True, "message": "Training and evaluation completed successfully", "details": message}
        elif success is False:
            return {"Success":False, "message": "Error occurred during training and evaluation", "details": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to classify text using a specified model
@app.post("/classify_text/")
def classify_text(data : ClassifyTextData):
    try:
        result = text_classifier.classify_text(data.datamodel_id, data.text)
        return {"classification_result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)