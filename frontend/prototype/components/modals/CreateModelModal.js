"use client";
import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.css";
import { Modal, Button, Form } from "react-bootstrap";

const CreateModelModal = ({
  show,
  onHide,
  classes,
  baseModels,
  createModelCallback,
}) => {
  const [modelName, setModelName] = useState("");
  const [selectedClasses, setSelectedClasses] = useState([]);
  const [dataEnrichmentEnabled, setDataEnrichmentEnabled] = useState(false);
  const [selectedBaseModels, setSelectedBaseModels] = useState([]);

  const handleInputChange = (event) => {
    setModelName(event.target.value);
  };

  const handleClassesCheckboxChange = (classId) => {
    // Toggle the selection of the class
    setSelectedClasses((prevSelectedClasses) => {
      if (prevSelectedClasses.includes(classId)) {
        return prevSelectedClasses.filter((id) => id !== classId);
      } else {
        return [...prevSelectedClasses, classId];
      }
    });
  };

  const handleDataEnrichmentCheckBoxChange = () => {
    // Toggle the data enrichment selection
    setDataEnrichmentEnabled(!dataEnrichmentEnabled);
  };

  const handleBaseModelsCheckBoxChange = (modelName) => {
    // Toggle the selection of the base model
    setSelectedBaseModels((prevSelectedBaseModels) => {
      if (prevSelectedBaseModels.includes(modelName)) {
        return prevSelectedBaseModels.filter((model) => model !== modelName);
      } else {
        return [...prevSelectedBaseModels, modelName];
      }
    });
  };

  const handleSubmitClick = () => {
    if (!modelName || !selectedClasses.length || !selectedBaseModels.length) {
      alert("Please Provide Model name, Classes and Base models.");
      return;
    }
    // Create the data object
    const modelData = {
      model_name: modelName,
      class_name_list: selectedClasses,
      selected_models: selectedBaseModels,
    };

    createModelCallback(modelData);
    onHide();
  };

  return (
    <Modal show={show} onHide={onHide} size="md">
      <Modal.Header closeButton>
        <Modal.Title>Create Model</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form>
          <Form.Group controlId="modelName" className="mx-5 my-4">
            <Form.Label>Model Name:</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter model name"
              value={modelName}
              onChange={handleInputChange}
              required
            />
          </Form.Group>
          <div className="mx-5">
            <p className="text-gray-600 mb-4 font-bold">Classes</p>
            <div className="max-h-[100px] overflow-y-auto m-4">
              {classes?.map((item, index) => (
                <Form.Check
                  key={index}
                  type="checkbox"
                  id={`${index}-${item}`}
                  label={item}
                  checked={selectedClasses.includes(item)}
                  onChange={() => handleClassesCheckboxChange(item)}
                  className="px-5 capitalize"
                />
              ))}
            </div>
            <div className="flex justify-start">
              <Form.Check
                type="checkbox"
                id={"dataEnrichment"}
                label={"Enable Data Enrichment ?"}
                checked={dataEnrichmentEnabled}
                onChange={handleDataEnrichmentCheckBoxChange}
                className="font-bold"
                reverse={true}
                disabled
              />
            </div>
            <div className="my-4">
              <p className="text-gray-600 mb-4 font-bold">Base Models</p>
              {baseModels?.map((item) => (
                <Form.Check
                  key={item?.model_id}
                  type="checkbox"
                  id={`baseModel-${item?.model_id}`}
                  label={item?.model_label}
                  checked={selectedBaseModels.includes(item?.model_name)}
                  onChange={() =>
                    handleBaseModelsCheckBoxChange(item?.model_name)
                  }
                  className="px-5"
                />
              ))}
            </div>
          </div>
        </Form>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
        <Button variant="primary" onClick={handleSubmitClick}>
          Create & Train
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default CreateModelModal;
