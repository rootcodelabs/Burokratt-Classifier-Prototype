"use client";
import axios from "axios";
import React, { useState } from "react";
import { Modal, Button, Form } from "react-bootstrap";

const UploadDatasetModal = ({ show, onHide, onDatasetUploaded }) => {
  const [datasetName, setDatasetName] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  const handleInputChange = (event) => {
    setDatasetName(event.target.value);
  };

  const handleFileSubmit = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleSubmitClick = () => {
    if (!datasetName || !selectedFile) {
      alert("Please Provide a Dataset Name and a Dataset");
      return;
    }

    // Create the data object
    const data = {
      dataset_name: datasetName,
      file: selectedFile,
    };
    // Add any additional logic or API calls here
    axios
      .post("http://0.0.0.0:8000/datasets/import/", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        alert("Dataset Successfully Uploaded!");
        onDatasetUploaded();
        onHide();
      })
      .catch((error) => {
        // Handle errors
        console.error("Error posting data:", error);
        alert(error?.message);
        onHide();
      });
  };

  return (
    <Modal show={show} onHide={onHide} size="md">
      <Modal.Header closeButton>
        <Modal.Title>Upload Dataset</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form>
          <Form.Group controlId="datasetName" className="mx-5 my-4">
            <Form.Label>Dataset Name:</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter Dataset name"
              value={datasetName}
              onChange={handleInputChange}
              required
            />
          </Form.Group>
          <Form.Group controlId="fileUpload" className="mx-5 my-4 ">
            <Form.Label>Import Dataset:</Form.Label>
            <Form.Control type="file" onChange={handleFileSubmit} required />
          </Form.Group>
        </Form>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
        <Button variant="primary" onClick={handleSubmitClick}>
          Upload Dataset
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default UploadDatasetModal;
