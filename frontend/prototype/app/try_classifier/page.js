"use client";
import Loader from "@/components/loaders/ScaleLoader";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.css";
import React, { useEffect, useState } from "react";
import { Button, Container, Form } from "react-bootstrap";
function TryClassifier() {
  const [textInput, setTextInput] = useState("");
  const [availableModels, setAvailableModels] = useState([]);

  const [classifierOutput, setClassifierOutput] = useState("");
  const [selectedModelName, setSelectedModelName] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");

  const [isDataLoading, setIsDataLoading] = useState(true);
  const [isClassifierLoading, setIsClassifierLoading] = useState(false);

  const handleInputChange = (event) => {
    setClassifierOutput("");
    setTextInput(event.target.value);
  };

  const handleModelChange = (e) => {
    const optionSelected = e.target.value;
    const selectedModelData = availableModels?.find(
      (model) => model.model_id === optionSelected
    );
    setSelectedModelName(selectedModelData?.model_name);
    setSelectedModelId(optionSelected);
  };

  const handleClassify = (e) => {
    // Handle class creation logic here
    e.preventDefault();
    if (!textInput || !selectedModelId) {
      alert("Please select a model & input the text to be classified");
      return;
    }
    const requestObject = {
      datamodel_id: selectedModelId,
      text: textInput,
    };
    setIsClassifierLoading(true);
    axios
      .post(`http://0.0.0.0:8000/classify_text`, requestObject)
      .then((response) => {
        // Process the response
        setClassifierOutput(response?.data?.classification_result);
        setIsClassifierLoading(false);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        alert("Error!");
        setIsClassifierLoading(false);
      });
  };

  const getAvailableModels = () => {
    setIsDataLoading(true);
    axios
      .get("http://0.0.0.0:8000/model_names/")
      .then((response) => {
        // Process the response
        setAvailableModels(response?.data);
        setIsDataLoading(false);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error fetching data:", error);
        alert("Error!");
        setIsDataLoading(false);
      });
  };

  useEffect(() => {
    getAvailableModels();
  }, []);

  return (
    <Container fluid>
      <div className="flex flex-col">
        <h2 className="text-center mt-5 mb-4">Try out the Classifier</h2>
        {isDataLoading ? (
          <Loader />
        ) : (
          <>
            <Form className="mt-4 lg:px-80">
              <Form.Group
                controlId="className"
                className="flex flex-row items-center mx-5 my-3"
              >
                <Form.Label className="basis-5/6">
                  Select Model for the Classifier:
                </Form.Label>
                <Form.Select
                  value={selectedModelId}
                  onChange={handleModelChange}
                >
                  <option disabled={true} value="">
                    --Choose a Model--
                  </option>
                  {availableModels?.map((model) => (
                    <option key={model?.model_id} value={model?.model_id}>
                      {model?.model_name}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
              <Form.Group controlId="className" className="mx-5 my-3">
                <Form.Label>Add your text here:</Form.Label>
                <Form.Control
                  as="textarea"
                  placeholder="Text input will be here"
                  value={textInput}
                  onChange={handleInputChange}
                  style={{ height: "300px" }}
                  required
                />
              </Form.Group>
              <div className="flex justify-center">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleClassify}
                  type="submit"
                  disabled={isClassifierLoading}
                >
                  Classify Text
                </Button>
              </div>
            </Form>
            <h5 className="text-center mt-4">{`This text belongs to the class : ${classifierOutput}`}</h5>
          </>
        )}
      </div>
    </Container>
  );
}

export default TryClassifier;
