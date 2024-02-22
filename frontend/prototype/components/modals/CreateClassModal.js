"use client";
import React, { useRef, useState } from "react";
import "bootstrap/dist/css/bootstrap.css";
import { Modal, Button, Form } from "react-bootstrap";
import axios from "axios";

const CreateClassModal = ({ show, onHide, onSuccessCallBack }) => {
  const [className, setClassName] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [textFields, setTextFields] = useState([""]); // Initialize with one empty field
  const [selectedFile, setSelectedFile] = useState(null);

  const handleAddTextField = () => {
    setTextFields([...textFields, ""]); // Add a new empty field
  };

  const handleRemoveTextField = (index) => {
    const newTextFields = [...textFields]; // Make a copy
    newTextFields.splice(index, 1); // Remove the field at the specified index
    setTextFields(newTextFields); // Update the state
  };

  const handleTextFieldChange = (index, value) => {
    const newTextFields = [...textFields]; // Make a copy
    newTextFields[index] = value; // Update the value at the specified index
    setTextFields(newTextFields); // Update the state
  };

  const handleFileSubmit = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleCreateClassOptionA = () => {
    if (!className || !datasetName || !selectedFile) {
      alert("Please provide the Class Name, Dataset Name and the Dataset");
      return;
    }

    const importedDatasetObject = {
      class_name: className,
      dataset_name: datasetName,
      file: selectedFile,
    };
    axios
      .post(
        `http://23.20.183.202:8000/class/datasets/import/`,
        importedDatasetObject,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      )
      .then((response) => {
        if (response?.status === 200) {
          alert("Class Successfully Created!");
          onSuccessCallBack();
          onHide();
        }
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        alert("Error!");
        onHide();
      });
  };

  const handleCreateClassOptionB = () => {
    const userData = textFields.filter((field) => field.trim() !== ""); // Remove empty fields

    if (!className) {
      alert("Please enter the Class Name");
      return;
    }
    if (!userData.length) {
      alert("Please enter at least one data field.");
      return;
    }

    const dataSamplesObject = {
      class_name: className,
      data: userData,
    };
    axios
      .post(`http://23.20.183.202:8000/class/data/add`, dataSamplesObject)
      .then((response) => {
        if (response?.status === 200) {
          alert("Class Successfully Created!");
          onSuccessCallBack();
          onHide();
        }
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        alert("Error!");
        onHide();
      });
  };

  return (
    <Modal show={show} onHide={onHide} size="md">
      <Modal.Header closeButton>
        <Modal.Title>Create Class</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form className="mx-5">
          <Form.Group controlId="className" className="my-2">
            <Form.Label>Class Name:</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter class name"
              value={className}
              onChange={(e) => setClassName(e.target.value)}
              required
            />
          </Form.Group>

          <div className="flex justify-center m-2 font-bold">
            Proceed by Importing a Dataset
          </div>

          <Form.Group
            controlId="datasetName"
            className="flex  flex-row items-center my-4"
          >
            <Form.Label className="basis-3/5">Dataset Name:</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter Dataset name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className="h-8"
              required
            />
          </Form.Group>
          <Form.Group
            controlId="fileUpload"
            className="flex flex-row  items-center my-4 "
          >
            <Form.Label className="basis-3/5">Dataset:</Form.Label>
            <Form.Control
              type="file"
              onChange={handleFileSubmit}
              required
              className="h-8"
            />
          </Form.Group>

          <div className="flex justify-center mt-2">
            <Button
              variant="primary"
              className="mt-2"
              onClick={handleCreateClassOptionA}
            >
              Create Class
            </Button>
          </div>
        </Form>
        <hr />
        <div className="flex justify-center my-4 font-bold">
          Or Add Data Samples Manually
        </div>
        <Form className="mx-5">
          <div className="mt-4">
            {textFields.map((textField, index) => (
              <div key={index} className="flex gap-0.5 items-center mb-2">
                <Form.Control
                  type="text"
                  placeholder={`Enter Data Sample ${index + 1}`}
                  value={textField}
                  onChange={(e) => handleTextFieldChange(index, e.target.value)}
                />
                <Button
                  variant="secondary"
                  onClick={() => handleAddTextField()}
                >
                  +
                </Button>
                {index > 0 && (
                  <Button
                    variant="danger"
                    onClick={() => handleRemoveTextField(index)}
                  >
                    -
                  </Button>
                )}
              </div>
            ))}
            <div className="flex justify-center">
              <Button
                variant="primary"
                className="mt-2"
                onClick={handleCreateClassOptionB}
              >
                Create Class
              </Button>
            </div>
          </div>
        </Form>
      </Modal.Body>
      <Modal.Footer></Modal.Footer>
    </Modal>
  );
};

export default CreateClassModal;
