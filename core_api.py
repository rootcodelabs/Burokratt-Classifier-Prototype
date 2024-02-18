import uvicorn
import shutil
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database.database_connection import SQLiteDatabase
from data_processor import DataImporter 
from text_classifier import TextClassifier

app = FastAPI()
db = SQLiteDatabase()
data_importer = DataImporter()
text_classifier = TextClassifier()

class LabelInput(BaseModel):
    dataset_id: str
    data_id: str
    label: str

class FileLocationInput(BaseModel):
    file_location: str

class DataInput(BaseModel):
    dataset_id: str
    data: str
    label: str

class MultipleDataLabelInput(BaseModel):
    data: str
    label: str

class MultiDataInput(BaseModel):
    dataset_id: str
    data_inputs: List[MultipleDataLabelInput]

class ModelInfo(BaseModel):
    model_id: str
    accuracy: float
    f1_score: float
    dataset_id: str    

# Endpoint to retrieve information about datasets
@app.get("/datasets/info/")
def get_datasets_info():
    try:
        query = """
        SELECT dataset_id,
               COUNT(data_id) AS data_count,
               COUNT(CASE WHEN label = '' THEN 1 END) AS unlabeled_count,
               COUNT(CASE WHEN label != '' THEN 1 END) AS labeled_count
        FROM dataset_info
        GROUP BY dataset_id
        """
        result = db.execute_query(query)
        datasets_info = [{"dataset_id": row[0], "data_count": row[1], "unlabeled_count": row[2], "labeled_count": row[3]} for row in result]
        return datasets_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve information about a single dataset
@app.get("/datasets/{dataset_id}/")
def get_single_dataset(dataset_id: str):
    try:
        query = f"SELECT data_id, data, label FROM dataset_info WHERE dataset_id = '{dataset_id}'"
        result = db.execute_query(query)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_info = [{"data_id": row[0], "data": row[1], "label": row[2]} for row in result]
        return {"dataset_id": dataset_id, "data": dataset_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to add a single label
@app.post("/datasets/label/")
def add_label(label_input: LabelInput):
    try:
        query = f"UPDATE dataset_info SET label = '{label_input.label}' WHERE dataset_id = '{label_input.dataset_id}' AND data_id = '{label_input.data_id}'"
        db.execute_query(query)
        return {"message": "Label added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to import data from file
@app.post("/datasets/import/")
def import_data(file_location_input: FileLocationInput):
    try:
        data_importer.import_data_from_file(file_location_input.file_location)
        return {"message": "Data imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to delete a dataset
@app.delete("/datasets/delete/")
def delete_dataset(dataset_id: str):
    try:
        db.delete_record("dataset_info", dataset_id)
        return {"message": f"Dataset with dataset_id {dataset_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to add new data to the dataset
@app.post("/datasets/add_data/")
def add_data(data_input: DataInput):
    try:
        dataset_exists = db.execute_query(f"SELECT COUNT(*) FROM dataset_info WHERE dataset_id = '{data_input.dataset_id}'")
        if dataset_exists[0][0] == 0:
            raise HTTPException(status_code=404, detail=f"Dataset with dataset_id {data_input.dataset_id} does not exist.")

        last_data_id_query = f"SELECT MAX(CAST(data_id AS INTEGER)) FROM dataset_info WHERE dataset_id = '{data_input.dataset_id}'"
        last_data_id_result = db.execute_query(last_data_id_query)
        last_data_id = last_data_id_result[0][0] if last_data_id_result[0][0] is not None else 0

        new_data_id = str(int(last_data_id) + 1)

        new_data_record = {
            'dataset_id': data_input.dataset_id,
            'data_id': new_data_id,
            'data': data_input.data,
            'label': data_input.label
        }
        db.insert_record('dataset_info', new_data_record)

        return {"message": f"New data added to dataset {data_input.dataset_id} with data_id {new_data_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to add new data or multiple data with different labels to the dataset
@app.post("/datasets/add_multi_data/")
def add_multi_data(multi_data_input: MultiDataInput):
    try:
        for data_input in multi_data_input.data_inputs:
            dataset_exists = db.execute_query(f"SELECT COUNT(*) FROM dataset_info WHERE dataset_id = '{multi_data_input.dataset_id}'")
            if dataset_exists[0][0] == 0:
                raise HTTPException(status_code=404, detail=f"Dataset with dataset_id {multi_data_input.dataset_id} does not exist.")

            last_data_id_query = f"SELECT MAX(CAST(data_id AS INTEGER)) FROM dataset_info WHERE dataset_id = '{multi_data_input.dataset_id}'"
            last_data_id_result = db.execute_query(last_data_id_query)
            last_data_id = last_data_id_result[0][0] if last_data_id_result[0][0] is not None else 0

            new_data_id = str(int(last_data_id) + 1)

            new_data_record = {
                'dataset_id': multi_data_input.dataset_id,
                'data_id': new_data_id,
                'data': data_input.data,
                'label': data_input.label
            }
            db.insert_record('dataset_info', new_data_record)

        return {"message": f"New data added to dataset {multi_data_input.dataset_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to retrieve information about all data inside model_info table
@app.get("/model_info/")
def get_model_info():
    try:
        query = "SELECT model_id, accuracy, f1_score, dataset_id FROM model_info"
        model_info_records = db.execute_query(query)

        model_info_list = []
        for record in model_info_records:
            model_info_list.append(ModelInfo(model_id=record[0], accuracy=record[1], f1_score=record[2], dataset_id=record[3]))

        return model_info_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to delete a specific row in the model_info table and corresponding model folder
@app.delete("/model_info/{model_id}")
def delete_model_info(model_id: str):
    try:
        model_folder_path = f"models/{model_id}"

        model_exists = db.execute_query(f"SELECT COUNT(*) FROM model_info WHERE model_id = '{model_id}'")
        if model_exists[0][0] == 0:
            if os.path.exists(model_folder_path):
                shutil.rmtree(model_folder_path)
                raise HTTPException(status_code=404, detail=f"Model with model_id {model_id} does not exist. But model files were deleted.")
            else:
                raise HTTPException(status_code=404, detail=f"Model with model_id {model_id} does not exist on both database and model folder.")
        else:
            db.execute_query(f"DELETE FROM model_info WHERE model_id = '{model_id}'")

        if os.path.exists(model_folder_path):
            shutil.rmtree(model_folder_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model with model_id {model_id} does not exist on folder. But model files were deleted from database.")

        return {"message": f"Model with model_id {model_id} and its corresponding folder deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to classify text using a specified model
@app.post("/classify_text/")
def classify_text(model_id: str, text: str):
    try:
        result = text_classifier.classify_text(model_id, text)
        return {"classification_result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)